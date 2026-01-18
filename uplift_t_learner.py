import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBClassifier

print("="*70)
print("UPLIFT MODELING: T-LEARNER META-LEARNER APPROACH")
print("="*70)

# ==================== 1. LOAD PREPROCESSED DATA ====================
print("\n📥 Loading preprocessed data...")
df = pd.read_csv('hillstrom_modeling_ready.csv')
print(f"✅ Loaded {len(df):,} records")

# ==================== 2. PREPARE FEATURES AND OUTCOME ====================
print("\n🔧 Preparing features and outcome...")

# Get feature columns (exclude outcome, treatment, and segment)
feature_cols = [col for col in df.columns if col not in ['visit', 'is_treated', 'segment']]
X = df[feature_cols]
y = df['visit']
treatment = df['is_treated']

print(f"Features: {len(feature_cols)}")
print(f"Outcome (visit): {y.sum():,} positives out of {len(y):,} ({y.mean()*100:.2f}%)")

# ==================== 3. TRAIN/TEST SPLIT ====================
print("\n✂️  Splitting data (80% Train / 20% Test)...")

X_train, X_test, y_train, y_test, treatment_train, treatment_test = train_test_split(
    X, y, treatment, 
    test_size=0.20, 
    random_state=42, 
    stratify=treatment
)

print(f"Train set: {len(X_train):,} records")
print(f"Test set: {len(X_test):,} records")

# ==================== 4. T-LEARNER: MODEL A (CONTROL) ====================
print("\n" + "="*70)
print("📊 MODEL A: CONTROL GROUP (is_treated = 0)")
print("="*70)

# Filter control group in training set
control_mask_train = treatment_train == 0
X_train_control = X_train[control_mask_train]
y_train_control = y_train[control_mask_train]

print(f"Control training samples: {len(X_train_control):,}")
print(f"Control visit rate: {y_train_control.mean()*100:.2f}%")

# Train XGBoost on Control Group
print("\n🚀 Training Model A (Control)...")
model_control = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)
model_control.fit(X_train_control, y_train_control)
print("✅ Model A trained")

# ==================== 5. T-LEARNER: MODEL B (TREATMENT) ====================
print("\n" + "="*70)
print("📧 MODEL B: TREATMENT GROUP (is_treated = 1)")
print("="*70)

# Filter treatment group in training set
treatment_mask_train = treatment_train == 1
X_train_treatment = X_train[treatment_mask_train]
y_train_treatment = y_train[treatment_mask_train]

print(f"Treatment training samples: {len(X_train_treatment):,}")
print(f"Treatment visit rate: {y_train_treatment.mean()*100:.2f}%")

# Train XGBoost on Treatment Group
print("\n🚀 Training Model B (Treatment)...")
model_treatment = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)
model_treatment.fit(X_train_treatment, y_train_treatment)
print("✅ Model B trained")

# ==================== 6. PREDICTION & UPLIFT CALCULATION ====================
print("\n" + "="*70)
print("🎯 CALCULATING UPLIFT SCORES")
print("="*70)

# Predict probabilities for test set using both models
print("\n📈 Predicting P(visit | no_treatment) using Model A...")
p_control = model_control.predict_proba(X_test)[:, 1]  # Probability of visit=1

print("📈 Predicting P(visit | treatment) using Model B...")
p_treatment = model_treatment.predict_proba(X_test)[:, 1]  # Probability of visit=1

# Calculate Uplift Score
print("\n💡 Calculating Uplift Score = P(treatment) - P(control)...")
uplift_score = p_treatment - p_control

# ==================== 7. CREATE RESULTS DATAFRAME ====================
print("\n📊 Creating results dataframe...")

results_df = pd.DataFrame({
    'actual_treatment': treatment_test.values,
    'actual_visit': y_test.values,
    'P_control': p_control,
    'P_treatment': p_treatment,
    'uplift_score': uplift_score
})

# Add original features for context
results_df = pd.concat([
    results_df.reset_index(drop=True),
    X_test.reset_index(drop=True)
], axis=1)

# ==================== 8. DISPLAY RESULTS ====================
print("\n" + "="*70)
print("📋 TEST SET WITH UPLIFT SCORES (First 5 Rows)")
print("="*70)
print()

# Display key columns
display_cols = ['actual_treatment', 'actual_visit', 'P_control', 'P_treatment', 'uplift_score']
print(results_df[display_cols].head(5).to_string(index=True))

# ==================== 9. UPLIFT SUMMARY STATISTICS ====================
print("\n" + "="*70)
print("📊 UPLIFT SCORE SUMMARY STATISTICS")
print("="*70)

print(f"\nUplift Score Statistics:")
print(f"  Mean:     {uplift_score.mean():+.4f}")
print(f"  Median:   {np.median(uplift_score):+.4f}")
print(f"  Std Dev:  {uplift_score.std():.4f}")
print(f"  Min:      {uplift_score.min():+.4f}")
print(f"  Max:      {uplift_score.max():+.4f}")

# Categorize users
positive_uplift = (uplift_score > 0).sum()
negative_uplift = (uplift_score < 0).sum()
neutral_uplift = (uplift_score == 0).sum()

print(f"\nUplift Distribution:")
print(f"  Positive Uplift (Persuadables): {positive_uplift:,} ({positive_uplift/len(uplift_score)*100:.1f}%)")
print(f"  Negative Uplift (Do-Not-Disturb): {negative_uplift:,} ({negative_uplift/len(uplift_score)*100:.1f}%)")
print(f"  Neutral Uplift: {neutral_uplift:,} ({neutral_uplift/len(uplift_score)*100:.1f}%)")

# ==================== 10. SAVE RESULTS ====================
print("\n" + "="*70)
print("💾 SAVING RESULTS")
print("="*70)

results_df.to_csv('uplift_predictions_t_learner.csv', index=False)
print("✅ Saved: uplift_predictions_t_learner.csv")

print("\n" + "="*70)
print("✨ T-LEARNER UPLIFT MODEL COMPLETE")
print("="*70)

# ==================== 11. ADDITIONAL INSIGHTS ====================
print("\n" + "="*70)
print("💡 KEY INSIGHTS")
print("="*70)

# Top 10 users with highest uplift (most persuadable)
print("\n🎯 Top 5 Most Persuadable Users (Highest Uplift):")
top_uplift = results_df.nlargest(5, 'uplift_score')[display_cols]
print(top_uplift.to_string(index=True))

print("\n⚠️  Top 5 Do-Not-Disturb Users (Lowest/Most Negative Uplift):")
bottom_uplift = results_df.nsmallest(5, 'uplift_score')[display_cols]
print(bottom_uplift.to_string(index=True))

print("\n" + "="*70)
print("🎓 INTERPRETATION:")
print("="*70)
print("""
- Positive uplift_score: User benefits from treatment (send campaign)
- Negative uplift_score: Treatment hurts retention (don't send campaign)
- Near-zero uplift_score: Treatment has minimal effect (neutral)

Next Steps:
1. Target users with uplift_score > threshold for campaigns
2. Avoid users with negative uplift (Do-Not-Disturb segment)
3. Evaluate campaign ROI using uplift-based targeting
""")
