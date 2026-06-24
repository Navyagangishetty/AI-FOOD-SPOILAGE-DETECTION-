from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import json
import math
import os
import google.generativeai as genai

app = Flask(__name__)
CORS(app)

# ============================================================
# LOAD MODELS & ENCODERS
# ============================================================
print("\n[Loading Models...]")
try:
    rf_model = joblib.load('models/rf_model.pkl')
    xgb_model = joblib.load('models/xgb_model.pkl')
    le_category = joblib.load('models/label_encoder_category.pkl')
    le_label = joblib.load('models/label_encoder_label.pkl')
    le_season = joblib.load('models/label_encoder_season.pkl')
    scaler = joblib.load('models/scaler.pkl')
    features = joblib.load('models/features.pkl')
    
    with open('models/metadata.json', 'r') as f:
        metadata = json.load(f)
    
    with open('food_metadata.json', 'r') as f:
        food_metadata = json.load(f)
    
    # Load ensemble weights from metadata
    ensemble_weights = metadata.get('ensemble_weights', {'random_forest': 0.5, 'xgboost': 0.5})
    rf_weight = ensemble_weights['random_forest']
    xgb_weight = ensemble_weights['xgboost']
    
    print("[OK] All models loaded successfully")
    print(f"  - Features: {len(features)}")
    print(f"  - Classes: {metadata['classes']}")
    print(f"  - Ensemble weights: RF={rf_weight:.3f}, XGB={xgb_weight:.3f}")
    
except Exception as e:
    print(f"[ERROR] Error loading models: {e}")
    rf_model = None
    xgb_model = None
    metadata = {}
    food_metadata = {}
    rf_weight = 0.5
    xgb_weight = 0.5

# ============================================================
# GEN AI - FRESHNESS COUNSELOR CONFIG
# ============================================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "") # Set this in your environment
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
    print("[OK] Gemini AI Counselor initialized")
else:
    model = None
    print("[INFO] Gemini API Key missing. Counselor will use expert-system fallback.")

MOCK_ADVICE = {
    "CRITICAL": [
        "🚨 Critical Warning: Quality has dropped below safe retail levels. Redirect to processing (sauces/juices) or industrial use immediately to avoid total loss.",
        "🚨 Alert: High thermal stress detected. Quality is unsafe for standard markets. Consider animal feed or bio-compost conversion.",
        "🚨 Action Required: Spoilage triggers detected. Inspect batch immediately. Do not proceed with long-haul transit."
    ],
    "WARNING": [
        "⚠ Warning: Sub-optimal environment detected. Prioritize this batch for local markets or 'quick-sale' shelves.",
        "⚠ Advice: Environmental stress is mounting. Check container seals and cooling systems. Remaining life is shorter than predicted.",
        "⚠ Notice: Mild quality degradation. Still safe for consumption, but expedited delivery is recommended."
    ],
    "SAFE": [
        "✅ Status: Optimal conditions maintained. Food quality is excellent and suitable for premium retail or long-haul export.",
        "✅ Good News: Environment is perfect. No intervention required. Batch is highly stable.",
        "✅ Efficiency Check: Current storage parameters are effectively preserving shelf life. Keep it up!"
    ]
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_shelf_life_remaining(temp, base_shelf_life, transit_hours, t_opt_max):
    """
    Calculate remaining shelf life using simplified Arrhenius model
    """
    temp_stress = max(0, temp - t_opt_max)
    decay_factor = 1 + 0.08 * temp_stress  # Exponential decay approximation
    hours_consumed = transit_hours * decay_factor
    rsl = max(0, base_shelf_life - hours_consumed)
    return rsl, decay_factor

def calculate_food_quality_score(rf_prob_spoiled, xgb_prob_spoiled, 
                                  temp_dev, humidity_dev, shelf_life_pct):
    """
    Hybrid FQS calculation combining:
    - Model predictions
    - Environmental deviations
    - Shelf life consumption
    """
    # Model risk: average of both models' confidence in spoilage
    model_risk = (rf_prob_spoiled + xgb_prob_spoiled) / 2
    
    # Environmental risk
    temp_risk = min(0.3, temp_dev * 0.05)
    humidity_risk = min(0.2, humidity_dev * 0.02)
    
    # Shelf life risk
    shelf_risk = max(0, 1 - shelf_life_pct) * 0.2
    
    # Total risk (0-1)
    total_risk = min(1.0, model_risk * 0.5 + temp_risk + humidity_risk + shelf_risk)
    
    # FQS (0-100)
    fqs = (1 - total_risk) * 100
    return max(0, fqs)

def get_counselor_advice(food_name, status, fqs, rsl_hours, temp, humidity):
    """
    Get advice from Gemini or fallback to expert system
    """
    if model:
        try:
            prompt = f"""
            You are 'FreshGuard Counselor', an expert in food logistics and sustainability.
            Based on the following data, provide 2 short, professional sentences of advice.
            Focus on reducing waste and choosing the best destination.
            
            Food: {food_name}
            Status: {status}
            Quality Score: {fqs}/100
            Remaining Life: {rsl_hours} hours
            Current Conditions: {temp}°C, {humidity}% humidity
            
            If quality is low, suggest alternative routes like processing (sauce/juice), local sale, or compost.
            Keep it professional and concise. Don't use bullet points.
            """
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"GenAI Error: {e}")
            # Fallback will trigger below
    
    # Expert System Fallback
    import random
    choices = MOCK_ADVICE.get(status, MOCK_ADVICE["SAFE"])
    return random.choice(choices)

# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def index():
    categories = list(food_metadata.get('food_map', {}).keys())
    categories.sort()
    grouped_foods = food_metadata.get('food_map', {})
    
    # We should transform food_data to match what frontend expects
    food_data = {}
    for food, metrics in food_metadata.get('thresholds', {}).items():
        food_cat = next((cat for cat, items in grouped_foods.items() if food in items), "Unknown")
        food_data[food] = {
            "category": food_cat,
            "shelf_life_days": round(metrics.get("Base_Shelf_Life_Hours", 168) / 24, 1),
            "opt_temp": [metrics.get("Min_Optimal_Temp_C", 0), metrics.get("Max_Optimal_Temp_C", 10)],
            "opt_humidity": [metrics.get("Min_Optimal_Humidity_Pct", 50), metrics.get("Max_Optimal_Humidity_Pct", 90)]
        }
        
    return render_template("index.html", categories=categories, grouped_foods=grouped_foods, food_data=food_data)

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'models_loaded': rf_model is not None and xgb_model is not None,
        'ensemble': 'RF + XGBoost',
        'accuracy': f"{metadata.get('ensemble_accuracy', 0)*100:.2f}%",
        'version': metadata.get('version', 'unknown')
    })

@app.route('/api/categories', methods=['GET'])
def get_categories():
    """Get available food categories and items"""
    return jsonify(food_metadata['food_map'])

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint
    
    Request JSON:
    {
        "category": "Vegetable",
        "food_name": "Spinach",
        "temp": 20.0,
        "humidity": 60.0,
        "co2_ppm": 400.0,
        "ethylene_ppm": 0.0,
        "nh3_ppm": 0.0,
        "h2s_ppm": 0.0,
        "transit_hours": 8.0,
        "season": "Summer"
    }
    """
    try:
        data = request.json
        category = data.get('food_category', data.get('category'))
        food_name = data.get('food_type', data.get('food_name'))
        temp = float(data.get('temperature', data.get('temp', 25)))
        humidity = float(data.get('humidity', 60))
        co2_ppm = float(data.get('co2_ppm', 400))
        ethylene_ppm = float(data.get('ethylene_ppm', 0))
        nh3_ppm = float(data.get('nh3_ppm', 0))
        h2s_ppm = float(data.get('h2s_ppm', 0))
        transit_hours = float(data.get('transit_hrs', data.get('transit_hours', 24)))
        season = data.get('season', 'Summer')

        if not category and food_name:
            # Try to determine category from grouped_foods if missing
            grouped_foods = food_metadata.get('food_map', {})
            category = next((cat for cat, items in grouped_foods.items() if food_name in items), "Unknown")

        # Get food-specific thresholds
        item_thresholds = food_metadata['thresholds'].get(food_name, {})
        if not item_thresholds:
            item_thresholds = metadata.get('category_defaults', {}).get(category, {})

        t_opt_max = item_thresholds.get('Max_Optimal_Temp_C', 25)
        t_opt_min = item_thresholds.get('Min_Optimal_Temp_C', 0)
        h_opt_max = item_thresholds.get('Max_Optimal_Humidity_Pct', 80)
        h_opt_min = item_thresholds.get('Min_Optimal_Humidity_Pct', 40)
        base_shelf_life = item_thresholds.get('Base_Shelf_Life_Hours', 168)

        # Feature engineering (MUST match training)
        temp_deviation = (max(0, temp - t_opt_max) + 
                         max(0, t_opt_min - temp))
        humidity_deviation = (max(0, humidity - h_opt_max) + 
                              max(0, h_opt_min - humidity))
        
        env_stress = 1 + 0.08 * max(0, temp - t_opt_max)
        hours_consumed = transit_hours * env_stress
        remaining_shelf_life = max(0, base_shelf_life - hours_consumed)
        shelf_life_ratio = remaining_shelf_life / (base_shelf_life + 1e-6)

        # Log-transformed gas levels
        log_co2 = np.log1p(co2_ppm)
        log_ethylene = np.log1p(ethylene_ppm)
        log_nh3 = np.log1p(nh3_ppm)
        log_h2s = np.log1p(h2s_ppm)

        # Encode category
        try:
            cat_encoded = le_category.transform([category])[0]
        except ValueError:
            cat_encoded = 0  # Fallback for unknown categories

        # Encode season
        try:
            season_encoded = le_season.transform([season])[0]
        except ValueError:
            season_encoded = 0  # Fallback for unknown seasons

        # Feature vector (10 features — MUST match training FEATURES list)
        feature_vector = np.array([
            temp,                                      # Temperature_C
            humidity,                                  # Humidity_Pct
            transit_hours,                             # Transit_Duration_Hours
            t_opt_min,                                 # Min_Optimal_Temp_C
            t_opt_max,                                 # Max_Optimal_Temp_C
            h_opt_min,                                 # Min_Optimal_Humidity_Pct
            h_opt_max,                                 # Max_Optimal_Humidity_Pct
            base_shelf_life,                           # Base_Shelf_Life_Hours
            cat_encoded,                               # category_encoded
            season_encoded,                            # season_encoded
        ])

        # Scale features
        feature_vector_scaled = scaler.transform([feature_vector])[0]

        # ─── MODEL PREDICTIONS ───
        rf_pred = rf_model.predict([feature_vector_scaled])[0]
        rf_proba = rf_model.predict_proba([feature_vector_scaled])[0]
        
        xgb_pred = xgb_model.predict([feature_vector_scaled])[0]
        xgb_proba = xgb_model.predict_proba([feature_vector_scaled])[0]

        # Ensemble prediction (performance-weighted)
        ensemble_proba = rf_weight * rf_proba + xgb_weight * xgb_proba
        ensemble_pred = np.argmax(ensemble_proba)

        # Probability of spoilage (CRITICAL class)
        spoiled_idx = list(le_label.classes_).index('Spoiled') if 'Spoiled' in le_label.classes_ else -1
        prob_spoilage = ensemble_proba[spoiled_idx] if spoiled_idx != -1 else 0.0

        # Decode prediction
        ensemble_status = le_label.classes_[ensemble_pred]
        ensemble_confidence = float(ensemble_proba[ensemble_pred]) * 100

        # ─── QUALITY SCORING ───
        rsl, decay_factor = calculate_shelf_life_remaining(
            temp, base_shelf_life, transit_hours, t_opt_max
        )
        
        fqs = calculate_food_quality_score(
            rf_proba[spoiled_idx] if spoiled_idx != -1 else 0.1,
            xgb_proba[spoiled_idx] if spoiled_idx != -1 else 0.1,
            temp_deviation,
            humidity_deviation,
            shelf_life_ratio
        )

        # ─── STATUS MAPPING ───
        if fqs >= 75:
            status, color = "SAFE", "green"
        elif fqs >= 50:
            status, color = "WARNING", "yellow"
        else:
            status, color = "CRITICAL", "red"

        # Alert message
        alert = ""
        if prob_spoilage > 0.6:
            alert = "⚠️ High spoilage risk detected. Expedite delivery or inspect."
        elif fqs < 50:
            alert = "⚠️ Quality degradation detected. Monitor closely."

        return jsonify({
            'food_name': food_name,
            'category': category,
            'status': status,
            'color': color,
            'fqs': round(fqs, 1),
            'rsl_hours': round(rsl, 1),
            'confidence': round(ensemble_confidence, 1),
            'spoilage_risk': round(prob_spoilage * 100, 1),
            'rf_pred': le_label.classes_[rf_pred],
            'xgb_pred': le_label.classes_[xgb_pred],
            'ensemble_pred': ensemble_status,
            'decay_factor': round(decay_factor, 2),
            'alert': alert,
            'counselor_advice': get_counselor_advice(food_name, status, fqs, rsl, temp, humidity),
            'optimal_ranges': {
                'temp': [t_opt_min, t_opt_max],
                'humidity': [h_opt_min, h_opt_max]
            }
        })

    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    print("\n" + "="*70)
    print("   FreshGuard Backend API v4")
    print("="*70)
    print(f"\nEnsemble: Random Forest ({rf_weight:.1%}) + XGBoost ({xgb_weight:.1%})")
    print(f"Accuracy: {metadata.get('ensemble_accuracy', 0)*100:.2f}%")
    print(f"Features: {len(features)}")
    print("\nStarting server on http://localhost:5000")
    print("="*70 + "\n")
    
    app.run(debug=True, port=5000, threaded=True)