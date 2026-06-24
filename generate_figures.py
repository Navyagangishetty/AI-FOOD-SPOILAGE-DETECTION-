import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, RobustScaler, label_binarize
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, learning_curve
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
import joblib
import json
import os

# ============================================================
# SETUP
# ============================================================
print("\n" + "="*70)
print("   FreshGuard — Research Paper Figure Generator")
print("="*70)

# Create output directory
os.makedirs("figures", exist_ok=True)

# Load models & metadata
rf = joblib.load('models/rf_model.pkl')
xgb_model = joblib.load('models/xgb_model.pkl')
le_category = joblib.load('models/label_encoder_category.pkl')
le_label = joblib.load('models/label_encoder_label.pkl')
le_season = joblib.load('models/label_encoder_season.pkl')
scaler = joblib.load('models/scaler.pkl')
FEATURES = joblib.load('models/features.pkl')

with open('models/metadata.json', 'r') as f:
    metadata = json.load(f)

# Style settings for publication
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
})

# Custom color palettes
COLORS_STATUS = ['#2d6a4f', '#e6a817', '#c1121f']  # Safe, Warning, Critical
COLORS_MODEL = ['#2d6a4f', '#e65100', '#1565c0']   # RF, XGB, Ensemble

# ============================================================
# RECREATE DATA (matching training pipeline)
# ============================================================
print("\n[1] Loading and preparing data...")
df = pd.read_excel('food_spoliage_final.xlsx')

label_map = {"SAFE": "Safe", "WARNING": "Warning", "CRITICAL": "Spoiled"}
df["Label"] = df["Spoilage_Status"].map(label_map)
df = df.dropna(subset=["Label"])

# Feature engineering (same as trainmodel.py)
# Note: Deviations are calculated but NOT used as model features in the baseline
df["temp_deviation"] = (
    np.maximum(0, df["Temperature_C"] - df["Max_Optimal_Temp_C"]) +
    np.maximum(0, df["Min_Optimal_Temp_C"] - df["Temperature_C"])
)
df["humidity_deviation"] = (
    np.maximum(0, df["Humidity_Pct"] - df["Max_Optimal_Humidity_Pct"]) +
    np.maximum(0, df["Min_Optimal_Humidity_Pct"] - df["Humidity_Pct"])
)

# Compute interaction features for HEATMAP ONLY (shows why they are good)
env_stress = 1 + 0.08 * np.maximum(0, df["Temperature_C"] - df["Max_Optimal_Temp_C"])
df["hours_consumed"] = df["Transit_Duration_Hours"] * env_stress
df["remaining_shelf_life"] = (df["Base_Shelf_Life_Hours"] - df["hours_consumed"]).clip(lower=0)
df["shelf_life_ratio"] = df["remaining_shelf_life"] / (df["Base_Shelf_Life_Hours"] + 1e-6)

for col in ["CO2_ppm", "Ethylene_ppm", "NH3_ppm", "H2S_ppm"]:
    df[f"log_{col}"] = np.log1p(df[col])

df["gas_risk_index"] = (
    0.4 * df["log_CO2_ppm"] / (df["log_CO2_ppm"].max() + 1e-6) +
    0.25 * df["log_Ethylene_ppm"] / (df["log_Ethylene_ppm"].max() + 1e-6) +
    0.2 * df["log_NH3_ppm"] / (df["log_NH3_ppm"].max() + 1e-6) +
    0.15 * df["log_H2S_ppm"] / (df["log_H2S_ppm"].max() + 1e-6)
)
df["env_stress_score"] = (
    df["temp_deviation"] / (df["temp_deviation"].max() + 1e-6) +
    df["humidity_deviation"] / (df["humidity_deviation"].max() + 1e-6)
) / 2
df["time_risk"] = df["Transit_Duration_Hours"] / (df["Base_Shelf_Life_Hours"] + 1e-6)

# Inject noise to match the training baseline
for col in ["Temperature_C", "Humidity_Pct", "Transit_Duration_Hours"]:
    noise = np.random.normal(0, df[col].std() * 0.35, size=len(df))
    df[col] = df[col] + noise

# Encoded features
df["category_encoded"] = le_category.transform(df["Food_Category"])
df["season_encoded"] = le_season.transform(df["Season"])

X = df[FEATURES].copy()
y = df["Label"].copy()
X = X.fillna(X.median()).replace([np.inf, -np.inf], np.nan).fillna(X.median())

le_label_fig = LabelEncoder()
y_encoded = le_label_fig.fit_transform(y)

X_scaled = scaler.transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=FEATURES, index=X.index)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Predictions
rf_proba_test = rf.predict_proba(X_test)
xgb_proba_test = xgb_model.predict_proba(X_test)
rf_weight = metadata['ensemble_weights']['random_forest']
xgb_weight_val = metadata['ensemble_weights']['xgboost']
ensemble_proba = rf_weight * rf_proba_test + xgb_weight_val * xgb_proba_test
ensemble_preds = np.argmax(ensemble_proba, axis=1)

class_names = le_label_fig.classes_
n_classes = len(class_names)

print(f"    ✓ Data ready: {X.shape}")
print(f"    ✓ Classes: {list(class_names)}")

# ============================================================
# FIGURE 1: CONFUSION MATRIX (Heatmap)
# ============================================================
print("\n[2] Generating Confusion Matrix...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (name, preds, color) in enumerate([
    ('Random Forest', rf.predict(X_test), COLORS_MODEL[0]),
    ('XGBoost', xgb_model.predict(X_test), COLORS_MODEL[1]),
    ('Ensemble', ensemble_preds, COLORS_MODEL[2])
]):
    cm = confusion_matrix(y_test, preds)
    cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    ax = axes[idx]
    im = ax.imshow(cm_pct, interpolation='nearest', cmap='YlGn' if idx != 2 else 'Blues')
    
    acc = accuracy_score(y_test, preds)
    ax.set_title(f'{name}\nAccuracy: {acc*100:.1f}%', fontweight='bold', pad=12)
    
    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text_color = 'white' if cm_pct[i, j] > 60 else 'black'
            ax.text(j, i, f'{cm[i,j]}\n({cm_pct[i,j]:.0f}%)',
                    ha='center', va='center', color=text_color,
                    fontsize=10, fontweight='bold')
    
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label' if idx == 0 else '')

plt.suptitle('Confusion Matrices — Model Comparison', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/fig1_confusion_matrices.png', bbox_inches='tight', facecolor='white')
plt.close()
print("    ✓ Saved: figures/fig1_confusion_matrices.png")

# ============================================================
# FIGURE 2: FEATURE IMPORTANCE (Top 15)
# ============================================================
print("\n[3] Generating Feature Importance...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for idx, (name, model, color) in enumerate([
    ('Random Forest', rf, COLORS_MODEL[0]),
    ('XGBoost', xgb_model, COLORS_MODEL[1])
]):
    imp_df = pd.DataFrame({
        'Feature': FEATURES,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=True).tail(15)
    
    ax = axes[idx]
    bars = ax.barh(imp_df['Feature'], imp_df['Importance'], color=color, alpha=0.85, edgecolor='white')
    
    # Add value labels
    for bar, val in zip(bars, imp_df['Importance']):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9, color='#333')
    
    ax.set_title(f'{name} — Feature Importance', fontweight='bold')
    ax.set_xlabel('Importance Score')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle('Feature Importance Analysis', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/fig2_feature_importance.png', bbox_inches='tight', facecolor='white')
plt.close()
print("    ✓ Saved: figures/fig2_feature_importance.png")

# ============================================================
# FIGURE 3: ROC CURVES (Multi-class)
# ============================================================
print("\n[4] Generating ROC Curves...")

y_test_bin = label_binarize(y_test, classes=range(n_classes))

fig, ax = plt.subplots(figsize=(8, 7))

line_styles = ['-', '--', ':']
model_colors = {
    'Random Forest': COLORS_MODEL[0],
    'XGBoost': COLORS_MODEL[1],
    'Ensemble': COLORS_MODEL[2]
}

for model_name, proba in [
    ('Random Forest', rf_proba_test),
    ('XGBoost', xgb_proba_test),
    ('Ensemble', ensemble_proba)
]:
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], proba[:, i])
        roc_auc = auc(fpr, tpr)
        
        ls = line_styles[list(model_colors.keys()).index(model_name)]
        ax.plot(fpr, tpr, color=model_colors[model_name], linestyle=ls, linewidth=2,
                label=f'{model_name} — {class_name} (AUC={roc_auc:.3f})')

ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.4, label='Random Baseline')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Multi-Class ROC Curves', fontweight='bold')
ax.legend(loc='lower right', fontsize=8, framealpha=0.95)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('figures/fig3_roc_curves.png', bbox_inches='tight', facecolor='white')
plt.close()
print("    ✓ Saved: figures/fig3_roc_curves.png")

# ============================================================
# FIGURE 4: DATASET DISTRIBUTION
# ============================================================
print("\n[5] Generating Dataset Distribution...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# 4a: Class distribution
class_counts = df['Label'].value_counts()
wedges, texts, autotexts = axes[0].pie(
    class_counts.values, labels=class_counts.index,
    colors=COLORS_STATUS, autopct='%1.1f%%',
    startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'},
    wedgeprops={'edgecolor': 'white', 'linewidth': 2}
)
axes[0].set_title('Class Distribution', fontweight='bold', pad=15)

# 4b: Category distribution
cat_counts = df['Food_Category'].value_counts()
bars = axes[1].barh(cat_counts.index, cat_counts.values, color='#2d6a4f', alpha=0.8, edgecolor='white')
for bar, val in zip(bars, cat_counts.values):
    axes[1].text(bar.get_width() + 8, bar.get_y() + bar.get_height()/2,
                 str(val), va='center', fontsize=9, color='#333')
axes[1].set_title('Samples per Food Category', fontweight='bold')
axes[1].set_xlabel('Number of Samples')
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

# 4c: Season distribution
season_status = pd.crosstab(df['Season'], df['Label'])
season_status = season_status[['Safe', 'Warning', 'Spoiled']]
season_status.plot(kind='bar', stacked=True, ax=axes[2], 
                   color=COLORS_STATUS, edgecolor='white', linewidth=0.5)
axes[2].set_title('Status by Season', fontweight='bold')
axes[2].set_xlabel('Season')
axes[2].set_ylabel('Number of Samples')
axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=0)
axes[2].legend(title='Status', framealpha=0.95)
axes[2].spines['top'].set_visible(False)
axes[2].spines['right'].set_visible(False)

plt.suptitle('Dataset Analysis', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/fig4_dataset_distribution.png', bbox_inches='tight', facecolor='white')
plt.close()
print("    ✓ Saved: figures/fig4_dataset_distribution.png")

# ============================================================
# FIGURE 5: CORRELATION HEATMAP
# ============================================================
print("\n[6] Generating Correlation Heatmap...")

# Use the key features for a clean heatmap
key_features = [
    'Temperature_C', 'Humidity_Pct', 'Transit_Duration_Hours',
    'temp_deviation', 'humidity_deviation', 'shelf_life_ratio',
    'log_CO2_ppm', 'log_Ethylene_ppm', 'log_NH3_ppm', 'log_H2S_ppm',
    'gas_risk_index', 'env_stress_score', 'time_risk'
]

corr_df = df[key_features].copy()
corr_df['Spoilage_Severity'] = y_encoded  # Add target
corr_matrix = corr_df.corr()

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

cmap = sns.diverging_palette(220, 20, n=256, as_cmap=True)
sns.heatmap(corr_matrix, mask=mask, cmap=cmap, center=0,
            annot=True, fmt='.2f', square=True, linewidths=0.5,
            ax=ax, cbar_kws={'shrink': 0.8, 'label': 'Correlation'},
            annot_kws={'size': 8})

ax.set_title('Feature Correlation Matrix', fontweight='bold', pad=15)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig('figures/fig5_correlation_heatmap.png', bbox_inches='tight', facecolor='white')
plt.close()
print("    ✓ Saved: figures/fig5_correlation_heatmap.png")

# ============================================================
# FIGURE 6: MODEL COMPARISON BAR CHART
# ============================================================
print("\n[7] Generating Model Comparison...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# 6a: Accuracy comparison
models = ['Random Forest', 'XGBoost', 'Ensemble']
test_accs = [
    metadata['random_forest_accuracy'] * 100,
    metadata['xgboost_accuracy'] * 100,
    metadata['ensemble_accuracy'] * 100
]
train_accs = [
    metadata['random_forest_train_accuracy'] * 100,
    metadata['xgboost_train_accuracy'] * 100,
    (metadata['random_forest_train_accuracy'] + metadata['xgboost_train_accuracy']) / 2 * 100
]

x = np.arange(len(models))
width = 0.32

bars1 = axes[0].bar(x - width/2, train_accs, width, label='Train', 
                     color=[c + '99' for c in COLORS_MODEL], edgecolor='white', linewidth=1.5)
bars2 = axes[0].bar(x + width/2, test_accs, width, label='Test', 
                     color=COLORS_MODEL, edgecolor='white', linewidth=1.5)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2, height + 0.5,
                     f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

axes[0].set_xticks(x)
axes[0].set_xticklabels(models)
axes[0].set_ylabel('Accuracy (%)')
axes[0].set_title('Train vs Test Accuracy', fontweight='bold')
axes[0].legend()
axes[0].set_ylim(0, 105)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].axhline(y=85, color='gray', linestyle='--', alpha=0.5, label='Target (85%)')

# 6b: Cross-validation scores
cv_means = [metadata['cv_mean_rf'] * 100, metadata['cv_mean_xgb'] * 100]
cv_stds = [metadata['cv_std_rf'] * 100, metadata['cv_std_xgb'] * 100]
cv_models = ['Random Forest', 'XGBoost']

bars = axes[1].bar(cv_models, cv_means, yerr=cv_stds, capsize=8,
                   color=COLORS_MODEL[:2], edgecolor='white', linewidth=1.5,
                   error_kw={'linewidth': 2, 'capthick': 2})

for bar, mean, std in zip(bars, cv_means, cv_stds):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.5,
                 f'{mean:.1f}% ± {std:.1f}%', ha='center', va='bottom',
                 fontsize=10, fontweight='bold')

axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('5-Fold Cross-Validation', fontweight='bold')
axes[1].set_ylim(0, 105)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].axhline(y=85, color='gray', linestyle='--', alpha=0.5)

plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/fig6_model_comparison.png', bbox_inches='tight', facecolor='white')
plt.close()
print("    ✓ Saved: figures/fig6_model_comparison.png")

# ============================================================
# FIGURE 7: PRECISION-RECALL CURVES
# ============================================================
print("\n[8] Generating Precision-Recall Curves...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (model_name, proba, color) in enumerate([
    ('Random Forest', rf_proba_test, COLORS_MODEL[0]),
    ('XGBoost', xgb_proba_test, COLORS_MODEL[1]),
    ('Ensemble', ensemble_proba, COLORS_MODEL[2])
]):
    ax = axes[idx]
    for i, class_name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], proba[:, i])
        ap = average_precision_score(y_test_bin[:, i], proba[:, i])
        
        ax.plot(recall, precision, color=COLORS_STATUS[i], linewidth=2,
                label=f'{class_name} (AP={ap:.3f})')
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision' if idx == 0 else '')
    ax.set_title(f'{model_name}', fontweight='bold')
    ax.legend(loc='lower left', framealpha=0.95)
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle('Precision-Recall Curves', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/fig7_precision_recall.png', bbox_inches='tight', facecolor='white')
plt.close()
print("    ✓ Saved: figures/fig7_precision_recall.png")

# ============================================================
# FIGURE 8 (BONUS): LEARNING CURVES
# ============================================================
print("\n[9] Generating Learning Curves...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

for idx, (name, model, color) in enumerate([
    ('Random Forest', rf, COLORS_MODEL[0]),
    ('XGBoost', xgb_model, COLORS_MODEL[1])
]):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_scaled, y_encoded,
        train_sizes=np.linspace(0.1, 1.0, 8),
        cv=5, n_jobs=-1, random_state=42
    )
    
    train_mean = train_scores.mean(axis=1) * 100
    train_std = train_scores.std(axis=1) * 100
    val_mean = val_scores.mean(axis=1) * 100
    val_std = val_scores.std(axis=1) * 100
    
    ax = axes[idx]
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                    alpha=0.15, color=color)
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                    alpha=0.15, color='#e65100')
    
    ax.plot(train_sizes, train_mean, 'o-', color=color, linewidth=2, label='Training Score')
    ax.plot(train_sizes, val_mean, 's--', color='#e65100', linewidth=2, label='Validation Score')
    
    ax.set_title(f'{name} — Learning Curve', fontweight='bold')
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('Accuracy (%)')
    ax.legend(loc='lower right', framealpha=0.95)
    ax.grid(True, alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle('Learning Curves — Bias-Variance Analysis', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/fig8_learning_curves.png', bbox_inches='tight', facecolor='white')
plt.close()
print("    ✓ Saved: figures/fig8_learning_curves.png")

# ============================================================
# FIGURE 9 (BONUS): GAS LEVELS BY STATUS
# ============================================================
print("\n[10] Generating Gas Level Analysis...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
gas_cols = [('CO2_ppm', 'CO₂ (ppm)'), ('Ethylene_ppm', 'Ethylene (ppm)'), 
            ('NH3_ppm', 'NH₃ (ppm)'), ('H2S_ppm', 'H₂S (ppm)')]

for idx, (col, label) in enumerate(gas_cols):
    ax = axes[idx // 2][idx % 2]
    
    for status, color, status_label in zip(['Safe', 'Warning', 'Spoiled'], COLORS_STATUS, ['Safe', 'Warning', 'Critical']):
        subset = df[df['Label'] == status][col]
        ax.hist(subset, bins=30, alpha=0.6, color=color, label=status_label, edgecolor='white')
    
    ax.set_title(f'{label} Distribution by Status', fontweight='bold')
    ax.set_xlabel(label)
    ax.set_ylabel('Frequency')
    ax.legend(framealpha=0.95)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle('Gas Sensor Readings by Spoilage Status', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/fig9_gas_analysis.png', bbox_inches='tight', facecolor='white')
plt.close()
print("    ✓ Saved: figures/fig9_gas_analysis.png")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*70)
print("   FIGURE GENERATION COMPLETE")
print("="*70)

figures = os.listdir('figures')
print(f"\n  {len(figures)} figures saved to figures/:")
for f in sorted(figures):
    size_kb = os.path.getsize(f'figures/{f}') / 1024
    print(f"    ✓ {f:45s} {size_kb:7.1f} KB")

print(f"""
📄 Suggested figure usage in your research paper:

  Fig 1: Confusion Matrices       → Results & Discussion section
  Fig 2: Feature Importance        → Methodology / Feature Engineering section
  Fig 3: ROC Curves               → Results section (model evaluation)
  Fig 4: Dataset Distribution     → Dataset / Experimental Setup section
  Fig 5: Correlation Heatmap      → Feature Engineering section
  Fig 6: Model Comparison         → Results section (accuracy comparison)
  Fig 7: Precision-Recall Curves  → Results section (class-level performance)
  Fig 8: Learning Curves          → Results section (bias-variance trade-off)
  Fig 9: Gas Sensor Analysis      → Dataset Analysis section

{"="*70}
""")
