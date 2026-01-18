"""
META-LEARNER COMPARISON: S-Learner vs T-Learner vs X-Learner
Demonstrates systematic evaluation of different causal ML approaches
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, mean_squared_error
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
import time

print("="*80)
print("META-LEARNER COMPARISON FOR UPLIFT MODELING")
print("="*80)

# ==================== 1. LOAD DATA ====================
print("\n📥 Loading preprocessed data...")
df = pd.read_csv('hillstrom_modeling_ready.csv')

feature_cols = [col for col in df.columns if col not in ['visit', 'is_treated', 'segment']]
X = df[feature_cols].values
y = df['visit'].values
treatment = df['is_treated'].values

# Train/test split
X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(
    X, y, treatment, test_size=0.20, random_state=42, stratify=treatment
)

print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

# ==================== 2. S-LEARNER ====================
print("\n" + "="*80)
print("📊 S-LEARNER (Single Model)")
print("="*80)
print("Trains ONE model on combined data with treatment as a feature")

start_time = time.time()

# Create feature matrix with treatment indicator
X_train_s = np.column_stack([X_train, t_train])
X_test_s_control = np.column_stack([X_test, np.zeros(len(X_test))])
X_test_s_treated = np.column_stack([X_test, np.ones(len(X_test))])

# Train single model
s_learner = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)
s_learner.fit(X_train_s, y_train)

# Predict under both treatments
p_s_control = s_learner.predict_proba(X_test_s_control)[:, 1]
p_s_treated = s_learner.predict_proba(X_test_s_treated)[:, 1]
uplift_s = p_s_treated - p_s_control

s_time = time.time() - start_time

print(f"✅ S-Learner trained in {s_time:.2f}s")
print(f"   Mean uplift: {uplift_s.mean():+.4f}")
print(f"   Uplift std: {uplift_s.std():.4f}")

# ==================== 3. T-LEARNER ====================
print("\n" + "="*80)
print("📊 T-LEARNER (Two Models)")
print("="*80)
print("Trains TWO separate models: one for control, one for treatment")

start_time = time.time()

# Split by treatment
control_mask = t_train == 0
treated_mask = t_train == 1

# Model A: Control
t_learner_control = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)
t_learner_control.fit(X_train[control_mask], y_train[control_mask])

# Model B: Treatment
t_learner_treated = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)
t_learner_treated.fit(X_train[treated_mask], y_train[treated_mask])

# Predict uplift
p_t_control = t_learner_control.predict_proba(X_test)[:, 1]
p_t_treated = t_learner_treated.predict_proba(X_test)[:, 1]
uplift_t = p_t_treated - p_t_control

t_time = time.time() - start_time

print(f"✅ T-Learner trained in {t_time:.2f}s")
print(f"   Mean uplift: {uplift_t.mean():+.4f}")
print(f"   Uplift std: {uplift_t.std():.4f}")

# ==================== 4. X-LEARNER ====================
print("\n" + "="*80)
print("📊 X-LEARNER (Cross-Validated Meta-Learner)")
print("="*80)
print("Trains models to directly predict treatment effects using imputed counterfactuals")

start_time = time.time()

# Stage 1: Train response models (same as T-Learner)
x_model_control = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
x_model_treated = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)

x_model_control.fit(X_train[control_mask], y_train[control_mask])
x_model_treated.fit(X_train[treated_mask], y_train[treated_mask])

# Stage 2: Impute treatment effects
# For control group: estimate what would have happened if treated
tau_control = y_train[control_mask] - x_model_treated.predict_proba(X_train[control_mask])[:, 1]

# For treatment group: estimate what would have happened if not treated  
tau_treated = x_model_control.predict_proba(X_train[treated_mask])[:, 1] - y_train[treated_mask]

# Stage 3: Train models to predict treatment effects
x_tau_control = XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
x_tau_treated = XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)

x_tau_control.fit(X_train[control_mask], tau_control)
x_tau_treated.fit(X_train[treated_mask], tau_treated)

# Predict treatment effects
# Weight by propensity score (here assume balanced, use 0.5)
uplift_x = (x_tau_control.predict(X_test) + x_tau_treated.predict(X_test)) / 2

x_time = time.time() - start_time

print(f"✅ X-Learner trained in {x_time:.2f}s")
print(f"   Mean uplift: {uplift_x.mean():+.4f}")
print(f"   Uplift std: {uplift_x.std():.4f}")

# ==================== 5. CALCULATE QINI CURVES ====================
print("\n" + "="*80)
print("📈 QINI CURVE COMPARISON")
print("="*80)

def calculate_qini(uplift_scores, treatment_test, outcome_test, n_bins=100):
    """Calculate Qini curve"""
    # Sort by uplift score
    sorted_idx = np.argsort(-uplift_scores)
    treatment_sorted = treatment_test[sorted_idx]
    outcome_sorted = outcome_test[sorted_idx]
    
    n = len(treatment_sorted)
    fractions = np.linspace(0, 1, n_bins + 1)
    qini = []
    
    for frac in fractions:
        n_targeted = int(frac * n)
        if n_targeted == 0:
            qini.append(0)
            continue
            
        # Count outcomes in targeted population
        targeted_treated = treatment_sorted[:n_targeted] == 1
        targeted_control = treatment_sorted[:n_targeted] == 0
        
        n_treated = targeted_treated.sum()
        n_control = targeted_control.sum()
        
        if n_treated > 0 and n_control > 0:
            visits_treated = outcome_sorted[:n_targeted][targeted_treated].sum()
            visits_control = outcome_sorted[:n_targeted][targeted_control].sum()
            
            # Incremental gain
            gain = visits_treated - (visits_control * n_treated / n_control)
        else:
            gain = 0
            
        qini.append(gain)
    
    return fractions, np.array(qini)

# Calculate Qini curves for all models
fractions_s, qini_s = calculate_qini(uplift_s, t_test, y_test)
fractions_t, qini_t = calculate_qini(uplift_t, t_test, y_test)
fractions_x, qini_x = calculate_qini(uplift_x, t_test, y_test)

# Random baseline
overall_treated_rate = y_test[t_test == 1].mean()
overall_control_rate = y_test[t_test == 0].mean()
ate = overall_treated_rate - overall_control_rate
n_treated_total = (t_test == 1).sum()
qini_random = fractions_t * ate * n_treated_total

# Calculate Qini coefficients (AUUC - Area Under Uplift Curve)
auuc_s = trapezoid(qini_s, fractions_s)
auuc_t = trapezoid(qini_t, fractions_t)
auuc_x = trapezoid(qini_x, fractions_x)
auuc_random = trapezoid(qini_random, fractions_t)

qini_coef_s = (auuc_s - auuc_random) / auuc_random if auuc_random != 0 else 0
qini_coef_t = (auuc_t - auuc_random) / auuc_random if auuc_random != 0 else 0
qini_coef_x = (auuc_x - auuc_random) / auuc_random if auuc_random != 0 else 0

print(f"\n📊 Qini Coefficients (higher is better):")
print(f"   S-Learner: {qini_coef_s:.4f} (AUUC: {auuc_s:.2f})")
print(f"   T-Learner: {qini_coef_t:.4f} (AUUC: {auuc_t:.2f})")
print(f"   X-Learner: {qini_coef_x:.4f} (AUUC: {auuc_x:.2f})")

# ==================== 6. UPLIFT AT TOP K ====================
print("\n" + "="*80)
print("🎯 UPLIFT @ TOP K (Precision Metrics)")
print("="*80)

def calculate_uplift_at_k(uplift_scores, treatment_test, outcome_test, k_values=[0.1, 0.2, 0.5]):
    """Calculate actual uplift in top K% of predictions"""
    results = {}
    
    for k in k_values:
        # Get top K% by uplift score
        n_top = int(k * len(uplift_scores))
        top_idx = np.argsort(-uplift_scores)[:n_top]
        
        # Calculate actual treatment effect in top K
        top_treated = treatment_test[top_idx] == 1
        top_control = treatment_test[top_idx] == 0
        
        if top_treated.sum() > 0 and top_control.sum() > 0:
            treated_rate = outcome_test[top_idx][top_treated].mean()
            control_rate = outcome_test[top_idx][top_control].mean()
            actual_uplift = treated_rate - control_rate
        else:
            actual_uplift = 0
        
        results[k] = actual_uplift
    
    return results

k_values = [0.1, 0.2, 0.5]
uplift_at_k_s = calculate_uplift_at_k(uplift_s, t_test, y_test, k_values)
uplift_at_k_t = calculate_uplift_at_k(uplift_t, t_test, y_test, k_values)
uplift_at_k_x = calculate_uplift_at_k(uplift_x, t_test, y_test, k_values)

print(f"\n{'Model':<15} {'Top 10%':>12} {'Top 20%':>12} {'Top 50%':>12}")
print("-" * 55)
print(f"{'S-Learner':<15} {uplift_at_k_s[0.1]:>11.4f} {uplift_at_k_s[0.2]:>11.4f} {uplift_at_k_s[0.5]:>11.4f}")
print(f"{'T-Learner':<15} {uplift_at_k_t[0.1]:>11.4f} {uplift_at_k_t[0.2]:>11.4f} {uplift_at_k_t[0.5]:>11.4f}")
print(f"{'X-Learner':<15} {uplift_at_k_x[0.1]:>11.4f} {uplift_at_k_x[0.2]:>11.4f} {uplift_at_k_x[0.5]:>11.4f}")

# ==================== 7. COMPARISON SUMMARY ====================
print("\n" + "="*80)
print("📋 COMPREHENSIVE COMPARISON")
print("="*80)

comparison_df = pd.DataFrame({
    'Model': ['S-Learner', 'T-Learner', 'X-Learner'],
    'Training Time (s)': [s_time, t_time, x_time],
    'Qini Coefficient': [qini_coef_s, qini_coef_t, qini_coef_x],
    'AUUC': [auuc_s, auuc_t, auuc_x],
    'Uplift @ Top 10%': [uplift_at_k_s[0.1], uplift_at_k_t[0.1], uplift_at_k_x[0.1]],
    'Uplift @ Top 20%': [uplift_at_k_s[0.2], uplift_at_k_t[0.2], uplift_at_k_x[0.2]],
    'Mean Uplift': [uplift_s.mean(), uplift_t.mean(), uplift_x.mean()],
    'Std Uplift': [uplift_s.std(), uplift_t.std(), uplift_x.std()]
})

print(f"\n{comparison_df.to_string(index=False)}")

# Identify best model
best_qini_idx = comparison_df['Qini Coefficient'].idxmax()
best_top10_idx = comparison_df['Uplift @ Top 10%'].idxmax()

print(f"\n🏆 WINNER:")
print(f"   Best Qini Coefficient: {comparison_df.loc[best_qini_idx, 'Model']}")
print(f"   Best Uplift @ Top 10%: {comparison_df.loc[best_top10_idx, 'Model']}")

# ==================== 8. VISUALIZATIONS ====================
print("\n" + "="*80)
print("📊 CREATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Qini Curves
ax1 = axes[0, 0]
ax1.plot(fractions_s * 100, qini_s, 'b-', linewidth=2, label=f'S-Learner (Qini: {qini_coef_s:.4f})')
ax1.plot(fractions_t * 100, qini_t, 'g-', linewidth=2, label=f'T-Learner (Qini: {qini_coef_t:.4f})')
ax1.plot(fractions_x * 100, qini_x, 'orange', linewidth=2, label=f'X-Learner (Qini: {qini_coef_x:.4f})')
ax1.plot(fractions_t * 100, qini_random, 'r--', linewidth=2, label='Random Targeting')
ax1.set_xlabel('% Population Targeted', fontsize=11)
ax1.set_ylabel('Cumulative Incremental Gains', fontsize=11)
ax1.set_title('Qini Curve Comparison', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Uplift @ Top K
ax2 = axes[0, 1]
k_labels = ['Top 10%', 'Top 20%', 'Top 50%']
x_pos = np.arange(len(k_labels))
width = 0.25

s_values = [uplift_at_k_s[0.1], uplift_at_k_s[0.2], uplift_at_k_s[0.5]]
t_values = [uplift_at_k_t[0.1], uplift_at_k_t[0.2], uplift_at_k_t[0.5]]
x_values = [uplift_at_k_x[0.1], uplift_at_k_x[0.2], uplift_at_k_x[0.5]]

ax2.bar(x_pos - width, s_values, width, label='S-Learner', alpha=0.8)
ax2.bar(x_pos, t_values, width, label='T-Learner', alpha=0.8)
ax2.bar(x_pos + width, x_values, width, label='X-Learner', alpha=0.8)

ax2.set_xlabel('Targeting Level', fontsize=11)
ax2.set_ylabel('Actual Uplift', fontsize=11)
ax2.set_title('Uplift @ Top K Comparison', fontsize=12, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(k_labels)
ax2.legend()
ax2.grid(True, axis='y', alpha=0.3)

# Plot 3: Uplift Distribution
ax3 = axes[1, 0]
ax3.hist(uplift_s, bins=50, alpha=0.5, label='S-Learner', density=True)
ax3.hist(uplift_t, bins=50, alpha=0.5, label='T-Learner', density=True)
ax3.hist(uplift_x, bins=50, alpha=0.5, label='X-Learner', density=True)
ax3.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Uplift')
ax3.set_xlabel('Uplift Score', fontsize=11)
ax3.set_ylabel('Density', fontsize=11)
ax3.set_title('Uplift Score Distributions', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Model Metrics Comparison
ax4 = axes[1, 1]
metrics = ['Qini\nCoefficient', 'AUUC', 'Uplift@10%']
s_metrics = [qini_coef_s, auuc_s / 100, uplift_at_k_s[0.1] * 10]  # Scaled for visibility
t_metrics = [qini_coef_t, auuc_t / 100, uplift_at_k_t[0.1] * 10]
x_metrics = [qini_coef_x, auuc_x / 100, uplift_at_k_x[0.1] * 10]

x_pos = np.arange(len(metrics))
width = 0.25

ax4.bar(x_pos - width, s_metrics, width, label='S-Learner', alpha=0.8)
ax4.bar(x_pos, t_metrics, width, label='T-Learner', alpha=0.8)
ax4.bar(x_pos + width, x_metrics, width, label='X-Learner', alpha=0.8)

ax4.set_ylabel('Score (Normalized)', fontsize=11)
ax4.set_title('Model Performance Metrics', fontsize=12, fontweight='bold')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(metrics)
ax4.legend()
ax4.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('metalearner_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved: metalearner_comparison.png")
plt.close()

# ==================== 9. SAVE RESULTS ====================
comparison_df.to_csv('metalearner_comparison_results.csv', index=False)
print("✅ Saved: metalearner_comparison_results.csv")

# Save all uplift scores for further analysis
uplift_comparison_df = pd.DataFrame({
    'uplift_s_learner': uplift_s,
    'uplift_t_learner': uplift_t,
    'uplift_x_learner': uplift_x,
    'actual_treatment': t_test,
    'actual_outcome': y_test
})
uplift_comparison_df.to_csv('uplift_scores_all_models.csv', index=False)
print("✅ Saved: uplift_scores_all_models.csv")

print("\n" + "="*80)
print("✨ META-LEARNER COMPARISON COMPLETE")
print("="*80)

print(f"""
🎓 KEY TAKEAWAYS:

1. S-LEARNER (Single Model):
   - Simplest approach, fastest training
   - Treats treatment as just another feature
   - May miss heterogeneous treatment effects
   - Best when: Treatment effect is homogeneous across population

2. T-LEARNER (Two Models):
   - Separate models for control and treatment groups
   - Captures heterogeneous treatment effects better
   - More computationally expensive (2 models)
   - Best when: Treatment effects vary across customer segments

3. X-LEARNER (Meta-Learner):
   - Most sophisticated, uses imputed counterfactuals
   - Directly models treatment effects
   - Requires careful implementation
   - Best when: Imbalanced treatment/control or small sample sizes

RECOMMENDATION: {comparison_df.loc[best_qini_idx, 'Model']} performs best on this dataset
""")
