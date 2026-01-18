import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import requests
from io import StringIO

print("="*70)
print("HILLSTROM EMAIL MARKETING DATASET - UPLIFT MODEL PREPROCESSING")
print("="*70)

# ==================== 1. LOAD DATA ====================
print("\n📥 Loading Hillstrom Email Marketing Dataset...")

try:
    df = pd.read_csv('hillstrom.csv')
    print(f"✅ Dataset loaded: {len(df):,} records")
except FileNotFoundError:
    print("⚠️  Dataset not found. Downloading from MineThatData source...")
    
    # Direct download from Kevin Hillstrom's MineThatData challenge
    url = "https://blog.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
    
    try:
        print(f"   Downloading from MineThatData (official source)...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        df = pd.read_csv(StringIO(response.text))
        df.to_csv('hillstrom.csv', index=False)
        print(f"✅ Dataset downloaded and saved: {len(df):,} records")
    except Exception as e:
        print(f"❌ Download failed: {str(e)}")
        print("\n📋 MANUAL DOWNLOAD INSTRUCTIONS:")
        print("   1. Visit: https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html")
        print("   2. Download: Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv")
        print("   3. Rename to: hillstrom.csv")
        print("   4. Place in: C:\\Users\\infam\\ChurnPrevention\\")
        print("   5. Re-run this script")
        raise Exception("Could not download dataset. Please download manually (see instructions above).")

print(f"\nOriginal columns: {list(df.columns)}")
print(f"Shape: {df.shape}")

# ==================== 2. FOCUS ON VISIT OUTCOME ====================
print("\n" + "-"*70)
print("🎯 OUTCOME VARIABLE: 'visit' (Binary Retention Proxy)")
print("-"*70)

# Check if visit column exists
if 'visit' not in df.columns:
    print("❌ Error: 'visit' column not found in dataset")
    print(f"Available columns: {list(df.columns)}")
else:
    print(f"Visit distribution:\n{df['visit'].value_counts()}")
    print(f"Visit rate: {df['visit'].mean()*100:.2f}%")

# ==================== 3. TREATMENT ENGINEERING ====================
print("\n" + "-"*70)
print("💉 TREATMENT ENGINEERING")
print("-"*70)

# Check segment column
print(f"\nOriginal segments:\n{df['segment'].value_counts()}")

# Create binary treatment column
df['is_treated'] = df['segment'].apply(
    lambda x: 1 if x in ['Mens E-Mail', 'Womens E-Mail'] else 0
)

print(f"\nTreatment assignment:")
print(df['is_treated'].value_counts())
print(f"\nTreatment rate: {df['is_treated'].mean()*100:.2f}%")

# Create separate treatment indicators (for granular analysis)
df['treated_mens'] = (df['segment'] == 'Mens E-Mail').astype(int)
df['treated_womens'] = (df['segment'] == 'Womens E-Mail').astype(int)

# ==================== 4. FEATURE ENCODING ====================
print("\n" + "-"*70)
print("🔧 FEATURE ENCODING")
print("-"*70)

# Make a copy for modeling
df_model = df.copy()

# Identify categorical and numerical columns
categorical_cols = ['zip_code', 'channel', 'history_segment']
numerical_cols = ['recency', 'history', 'mens', 'womens', 'newbie']

print(f"\nCategorical features to encode: {categorical_cols}")
print(f"Numerical features: {numerical_cols}")

# One-hot encode categorical variables
print("\n🔄 One-hot encoding categorical variables...")

# Check which categorical columns exist
existing_categorical = [col for col in categorical_cols if col in df_model.columns]

if existing_categorical:
    df_encoded = pd.get_dummies(df_model, columns=existing_categorical, prefix=existing_categorical, drop_first=False)
    print(f"✅ Encoded {len(existing_categorical)} categorical columns")
else:
    df_encoded = df_model.copy()
    print("⚠️  No categorical columns found to encode")

# Ensure all numerical features are numeric
print("\n🔢 Ensuring numerical features are properly typed...")
for col in numerical_cols:
    if col in df_encoded.columns:
        df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce')

print(f"\nFinal feature set shape: {df_encoded.shape}")
print(f"Total features: {df_encoded.shape[1]}")

# ==================== 5. EXPLORATORY DATA ANALYSIS (EDA) ====================
print("\n" + "="*70)
print("📊 EXPLORATORY DATA ANALYSIS")
print("="*70)

# Calculate Average Treatment Effect (ATE)
print("\n🎯 AVERAGE TREATMENT EFFECT (ATE) ANALYSIS")
print("-"*70)

# Split by treatment
control_group = df[df['is_treated'] == 0]
treated_group = df[df['is_treated'] == 1]

# Calculate retention rates
control_retention = control_group['visit'].mean()
treated_retention = treated_group['visit'].mean()

# Calculate ATE
ate = treated_retention - control_retention
ate_pct = ate * 100

print(f"\n📉 CONTROL GROUP (No E-Mail):")
print(f"   Sample size: {len(control_group):,}")
print(f"   Visit rate: {control_retention*100:.2f}%")

print(f"\n📧 TREATED GROUP (Mens/Womens E-Mail):")
print(f"   Sample size: {len(treated_group):,}")
print(f"   Visit rate: {treated_retention*100:.2f}%")

print(f"\n{'='*70}")
print(f"🎯 AVERAGE TREATMENT EFFECT (ATE):")
print(f"   ATE = {ate:.4f} ({ate_pct:+.2f} percentage points)")
print(f"   Relative Lift = {(ate/control_retention)*100:+.2f}%")
print(f"{'='*70}")

if ate > 0:
    print("\n✅ POSITIVE EFFECT: Email campaigns increase visit rates")
elif ate < 0:
    print("\n⚠️  NEGATIVE EFFECT: Email campaigns decrease visit rates")
else:
    print("\n➖ NO EFFECT: Email campaigns have no impact on visit rates")

# ==================== 6. GRANULAR TREATMENT ANALYSIS ====================
print("\n" + "-"*70)
print("🔍 GRANULAR TREATMENT BREAKDOWN")
print("-"*70)

treatment_analysis = df.groupby('segment').agg({
    'visit': ['count', 'sum', 'mean']
}).round(4)

treatment_analysis.columns = ['Total_Users', 'Visits', 'Visit_Rate']
treatment_analysis['Visit_Rate_Pct'] = (treatment_analysis['Visit_Rate'] * 100).round(2)

print(f"\n{treatment_analysis}")

# Calculate individual treatment effects
mens_effect = df[df['segment'] == 'Mens E-Mail']['visit'].mean() - control_retention
womens_effect = df[df['segment'] == 'Womens E-Mail']['visit'].mean() - control_retention

print(f"\n📨 Individual Treatment Effects:")
print(f"   Mens E-Mail:   {mens_effect:+.4f} ({mens_effect*100:+.2f}pp)")
print(f"   Womens E-Mail: {womens_effect:+.4f} ({womens_effect*100:+.2f}pp)")

# ==================== 7. FEATURE SUMMARY ====================
print("\n" + "-"*70)
print("📋 FEATURE SUMMARY FOR MODELING")
print("-"*70)

# Prepare final feature list for XGBoost
feature_cols = [col for col in df_encoded.columns if col not in 
                ['visit', 'conversion', 'spend', 'segment', 'is_treated', 'treated_mens', 'treated_womens']]

print(f"\nFeatures ready for XGBoost: {len(feature_cols)}")
print(f"\nSample features:")
for i, col in enumerate(feature_cols[:10], 1):
    print(f"   {i}. {col}")
if len(feature_cols) > 10:
    print(f"   ... and {len(feature_cols) - 10} more")

# Check for missing values
missing_summary = df_encoded[feature_cols].isnull().sum()
if missing_summary.sum() > 0:
    print(f"\n⚠️  Missing values detected:")
    print(missing_summary[missing_summary > 0])
else:
    print(f"\n✅ No missing values in feature set")

# ==================== 8. SAVE PREPROCESSED DATA ====================
print("\n" + "-"*70)
print("💾 SAVING PREPROCESSED DATA")
print("-"*70)

# Save the encoded dataframe
df_encoded.to_csv('hillstrom_preprocessed.csv', index=False)
print("✅ Saved: hillstrom_preprocessed.csv")

# Save a modeling-ready version (features + outcome + treatment)
modeling_df = df_encoded[feature_cols + ['visit', 'is_treated', 'segment']].copy()
modeling_df.to_csv('hillstrom_modeling_ready.csv', index=False)
print("✅ Saved: hillstrom_modeling_ready.csv")

print("\n" + "="*70)
print("✨ PREPROCESSING COMPLETE - READY FOR UPLIFT MODELING")
print("="*70)
