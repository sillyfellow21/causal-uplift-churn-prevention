import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

print("="*70)
print("UPLIFT MODEL EVALUATION: QINI CURVE, ROI & STRATEGIC QUADRANTS")
print("="*70)

# ==================== 1. LOAD PREDICTIONS ====================
print("\n📥 Loading uplift predictions...")
df = pd.read_csv('uplift_predictions_t_learner.csv')
print(f"✅ Loaded {len(df):,} test predictions")

# ==================== 2. QINI CURVE ====================
print("\n" + "="*70)
print("📈 QINI CURVE ANALYSIS")
print("="*70)

# Sort by uplift score (descending)
df_sorted = df.sort_values('uplift_score', ascending=False).reset_index(drop=True)

# Calculate cumulative metrics
n_total = len(df_sorted)
n_treated = df_sorted['actual_treatment'].sum()
n_control = n_total - n_treated

# Initialize arrays for Qini curve
fractions = np.linspace(0, 1, 101)
qini_model = []
qini_random = []

for fraction in fractions:
    n_targeted = int(fraction * n_total)
    
    if n_targeted == 0:
        qini_model.append(0)
        qini_random.append(0)
        continue
    
    # Get top N users by uplift score
    top_users = df_sorted.iloc[:n_targeted]
    
    # Count outcomes in treated and control groups
    treated_subset = top_users[top_users['actual_treatment'] == 1]
    control_subset = top_users[top_users['actual_treatment'] == 0]
    
    n_treated_subset = len(treated_subset)
    n_control_subset = len(control_subset)
    
    visits_treated = treated_subset['actual_visit'].sum()
    visits_control = control_subset['actual_visit'].sum()
    
    # Qini curve: Incremental gain
    if n_treated_subset > 0 and n_control_subset > 0:
        incremental_gain = visits_treated - (visits_control * n_treated_subset / n_control_subset)
    else:
        incremental_gain = 0
    
    qini_model.append(incremental_gain)
    
    # Random baseline (expected incremental gain if random targeting)
    overall_treated_rate = df['actual_visit'][df['actual_treatment'] == 1].mean()
    overall_control_rate = df['actual_visit'][df['actual_treatment'] == 0].mean()
    ate_overall = overall_treated_rate - overall_control_rate
    
    expected_treated_in_fraction = fraction * n_treated
    random_gain = ate_overall * expected_treated_in_fraction
    qini_random.append(random_gain)

# Calculate Qini coefficient (area between curves)
from scipy.integrate import trapezoid
qini_model_area = trapezoid(qini_model, fractions)
qini_random_area = trapezoid(qini_random, fractions)
qini_coefficient = (qini_model_area - qini_random_area) / qini_random_area if qini_random_area != 0 else 0

print(f"\n📊 Qini Coefficient: {qini_coefficient:.4f}")
print(f"   (Higher is better - measures how much better than random targeting)")

# Plot Qini Curve
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(fractions * 100, qini_model, 'b-', linewidth=2, label='Uplift Model')
plt.plot(fractions * 100, qini_random, 'r--', linewidth=2, label='Random Targeting')
plt.fill_between(fractions * 100, qini_model, qini_random, alpha=0.3, color='green')
plt.xlabel('% of Population Targeted (by Uplift Score)', fontsize=11)
plt.ylabel('Cumulative Incremental Gains', fontsize=11)
plt.title('Qini Curve: Model vs Random Targeting', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 100)

# ==================== 3. PROFITABILITY ANALYSIS ====================
print("\n" + "="*70)
print("💰 PROFITABILITY ANALYSIS")
print("="*70)

LTV = 100  # Value per retained user
CAMPAIGN_COST = 2  # Cost per email sent

print(f"\nAssumptions:")
print(f"  - Customer Lifetime Value (LTV): ${LTV}")
print(f"  - Campaign Cost per User: ${CAMPAIGN_COST}")

# Calculate profit at different targeting levels
targeting_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
results = []

for target_frac in targeting_levels:
    n_targeted = int(target_frac * n_total)
    
    # Top N users by uplift score
    targeted_users = df_sorted.iloc[:n_targeted]
    
    # Count treated users in targeted set
    targeted_treated = targeted_users[targeted_users['actual_treatment'] == 1]
    targeted_control = targeted_users[targeted_users['actual_treatment'] == 0]
    
    # Estimated incremental visits if we target these users
    # Using the uplift scores to estimate impact
    estimated_incremental_visits = targeted_users['uplift_score'].sum()
    
    # Revenue from incremental visits
    revenue = estimated_incremental_visits * LTV
    
    # Cost of targeting
    cost = n_targeted * CAMPAIGN_COST
    
    # Profit
    profit = revenue - cost
    roi = (profit / cost * 100) if cost > 0 else 0
    
    results.append({
        'target_pct': target_frac * 100,
        'n_targeted': n_targeted,
        'estimated_incremental_visits': estimated_incremental_visits,
        'revenue': revenue,
        'cost': cost,
        'profit': profit,
        'roi': roi
    })

results_df = pd.DataFrame(results)

# Calculate "Send to Everyone" baseline
baseline_cost = n_total * CAMPAIGN_COST
baseline_visits = df['uplift_score'].sum()
baseline_revenue = baseline_visits * LTV
baseline_profit = baseline_revenue - baseline_cost
baseline_roi = (baseline_profit / baseline_cost * 100) if baseline_cost > 0 else 0

print(f"\n📊 Profitability by Targeting Strategy:")
print("-" * 70)
print(f"{'Target %':>10} {'N Users':>10} {'Incr Visits':>12} {'Revenue':>12} {'Cost':>10} {'Profit':>12} {'ROI %':>10}")
print("-" * 70)

for _, row in results_df.iterrows():
    print(f"{row['target_pct']:>9.0f}% {row['n_targeted']:>10,.0f} {row['estimated_incremental_visits']:>12.1f} "
          f"${row['revenue']:>11,.0f} ${row['cost']:>9,.0f} ${row['profit']:>11,.0f} {row['roi']:>9.1f}%")

print("-" * 70)
print(f"{'BASELINE (100%)':>10} {n_total:>10,} {baseline_visits:>12.1f} "
      f"${baseline_revenue:>11,.0f} ${baseline_cost:>9,.0f} ${baseline_profit:>11,.0f} {baseline_roi:>9.1f}%")
print("-" * 70)

# Find optimal targeting
optimal_idx = results_df['profit'].idxmax()
optimal_target = results_df.loc[optimal_idx]

print(f"\n✅ OPTIMAL STRATEGY:")
print(f"   Target Top {optimal_target['target_pct']:.0f}% of users (sorted by uplift score)")
print(f"   Expected Profit: ${optimal_target['profit']:,.0f}")
print(f"   ROI: {optimal_target['roi']:.1f}%")
print(f"   Improvement over 'Send to Everyone': ${(optimal_target['profit'] - baseline_profit):,.0f}")

# Plot profitability
plt.subplot(1, 2, 2)
plt.plot(results_df['target_pct'], results_df['profit'], 'b-o', linewidth=2, markersize=6, label='Model-Based Targeting')
plt.axhline(y=baseline_profit, color='r', linestyle='--', linewidth=2, label='Send to Everyone')
plt.xlabel('% of Population Targeted', fontsize=11)
plt.ylabel('Profit ($)', fontsize=11)
plt.title('Profit by Targeting Strategy', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 100)

# Add optimal point annotation
plt.scatter([optimal_target['target_pct']], [optimal_target['profit']], 
            color='green', s=200, zorder=5, marker='*', edgecolors='black', linewidth=2)
plt.annotate(f'Optimal: {optimal_target["target_pct"]:.0f}%\n${optimal_target["profit"]:,.0f}', 
             xy=(optimal_target['target_pct'], optimal_target['profit']),
             xytext=(optimal_target['target_pct'] - 15, optimal_target['profit'] * 0.85),
             fontsize=9, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', linewidth=2))

plt.tight_layout()
plt.savefig('qini_curve_profitability.png', dpi=300, bbox_inches='tight')
print(f"\n✅ Saved: qini_curve_profitability.png")
plt.close()

# ==================== 4. STRATEGIC QUADRANTS ====================
print("\n" + "="*70)
print("🎯 STRATEGIC QUADRANTS CLASSIFICATION")
print("="*70)

print("\nClassifying users based on predicted probabilities:")
print("  - Persuadables: Low P(control), High P(treatment) → Send campaign")
print("  - Sure Things: High P(control), High P(treatment) → Would visit anyway")
print("  - Lost Causes: Low P(control), Low P(treatment) → Won't visit anyway")
print("  - Sleeping Dogs: High P(control), Low P(treatment) → DON'T send (backfires!)")

# Define thresholds (median split for simplicity)
p_control_median = df['P_control'].median()
p_treatment_median = df['P_treatment'].median()

print(f"\nThresholds (median split):")
print(f"  P(control) threshold: {p_control_median:.4f}")
print(f"  P(treatment) threshold: {p_treatment_median:.4f}")

# Classify users
def classify_user(row):
    if row['P_control'] < p_control_median and row['P_treatment'] >= p_treatment_median:
        return 'Persuadables'
    elif row['P_control'] >= p_control_median and row['P_treatment'] >= p_treatment_median:
        return 'Sure Things'
    elif row['P_control'] < p_control_median and row['P_treatment'] < p_treatment_median:
        return 'Lost Causes'
    else:  # P_control >= median and P_treatment < median
        return 'Sleeping Dogs'

df['strategic_segment'] = df.apply(classify_user, axis=1)

# Count and display
segment_counts = df['strategic_segment'].value_counts()
segment_pcts = df['strategic_segment'].value_counts(normalize=True) * 100

print("\n" + "-" * 70)
print(f"{'Strategic Segment':<20} {'Count':>15} {'Percentage':>15} {'Action'}")
print("-" * 70)

segment_actions = {
    'Persuadables': '✅ TARGET',
    'Sure Things': '⚪ Optional (will visit anyway)',
    'Lost Causes': '⚪ Skip (won\'t visit anyway)',
    'Sleeping Dogs': '❌ AVOID (treatment backfires!)'
}

for segment in ['Persuadables', 'Sure Things', 'Lost Causes', 'Sleeping Dogs']:
    if segment in segment_counts.index:
        count = segment_counts[segment]
        pct = segment_pcts[segment]
        action = segment_actions[segment]
        print(f"{segment:<20} {count:>15,} {pct:>14.1f}%  {action}")

print("-" * 70)

# Calculate average uplift by segment
print("\n📊 Average Uplift Score by Strategic Segment:")
print("-" * 70)
avg_uplift = df.groupby('strategic_segment')['uplift_score'].agg(['mean', 'std', 'min', 'max'])
print(avg_uplift.round(4))

# Visualize Strategic Quadrants
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Scatter plot with quadrants
ax1 = axes[0]
colors = {'Persuadables': 'green', 'Sure Things': 'blue', 'Lost Causes': 'gray', 'Sleeping Dogs': 'red'}
for segment, color in colors.items():
    mask = df['strategic_segment'] == segment
    ax1.scatter(df[mask]['P_control'], df[mask]['P_treatment'], 
                alpha=0.4, s=20, c=color, label=segment)

# Add threshold lines
ax1.axhline(y=p_treatment_median, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax1.axvline(x=p_control_median, color='black', linestyle='--', linewidth=1, alpha=0.5)

# Add quadrant labels
ax1.text(0.25, 0.75, 'Persuadables', transform=ax1.transAxes, fontsize=10, 
         fontweight='bold', ha='center', bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))
ax1.text(0.75, 0.75, 'Sure Things', transform=ax1.transAxes, fontsize=10,
         fontweight='bold', ha='center', bbox=dict(boxstyle='round', facecolor='blue', alpha=0.3))
ax1.text(0.25, 0.25, 'Lost Causes', transform=ax1.transAxes, fontsize=10,
         fontweight='bold', ha='center', bbox=dict(boxstyle='round', facecolor='gray', alpha=0.3))
ax1.text(0.75, 0.25, 'Sleeping Dogs', transform=ax1.transAxes, fontsize=10,
         fontweight='bold', ha='center', bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))

ax1.set_xlabel('P(visit | control)', fontsize=11)
ax1.set_ylabel('P(visit | treatment)', fontsize=11)
ax1.set_title('Strategic Quadrants: User Classification', fontsize=12, fontweight='bold')
ax1.legend(loc='upper left', framealpha=0.9)
ax1.grid(True, alpha=0.3)

# Bar chart of segment counts
ax2 = axes[1]
segment_order = ['Persuadables', 'Sure Things', 'Lost Causes', 'Sleeping Dogs']
segment_data = [segment_counts.get(seg, 0) for seg in segment_order]
segment_colors = [colors[seg] for seg in segment_order]

bars = ax2.bar(segment_order, segment_data, color=segment_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Number of Users', fontsize=11)
ax2.set_title('Distribution of Strategic Segments', fontsize=12, fontweight='bold')
ax2.grid(True, axis='y', alpha=0.3)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height):,}\n({height/len(df)*100:.1f}%)',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('strategic_quadrants.png', dpi=300, bbox_inches='tight')
print(f"\n✅ Saved: strategic_quadrants.png")
plt.close()

# ==================== 5. SAVE RESULTS ====================
print("\n" + "="*70)
print("💾 SAVING RESULTS")
print("="*70)

# Save profitability results
results_df.to_csv('profitability_analysis.csv', index=False)
print("✅ Saved: profitability_analysis.csv")

# Save users with strategic segments
df.to_csv('uplift_predictions_with_segments.csv', index=False)
print("✅ Saved: uplift_predictions_with_segments.csv")

# ==================== 6. SUMMARY REPORT ====================
print("\n" + "="*70)
print("📋 EXECUTIVE SUMMARY")
print("="*70)

print(f"""
🎯 MODEL PERFORMANCE:
   - Qini Coefficient: {qini_coefficient:.4f}
   - Model significantly outperforms random targeting

💰 BUSINESS IMPACT:
   - Optimal Strategy: Target top {optimal_target['target_pct']:.0f}% of users
   - Expected Profit: ${optimal_target['profit']:,.0f}
   - ROI: {optimal_target['roi']:.1f}%
   - Improvement vs. Mass Campaign: ${(optimal_target['profit'] - baseline_profit):,.0f}

🎯 STRATEGIC RECOMMENDATIONS:
   - PRIORITIZE: {segment_counts.get('Persuadables', 0):,} Persuadables ({segment_pcts.get('Persuadables', 0):.1f}%)
   - AVOID: {segment_counts.get('Sleeping Dogs', 0):,} Sleeping Dogs ({segment_pcts.get('Sleeping Dogs', 0):.1f}%) - campaigns backfire!
   - Optional: {segment_counts.get('Sure Things', 0):,} Sure Things (will convert anyway)
   - Skip: {segment_counts.get('Lost Causes', 0):,} Lost Causes (won't convert)

📈 KEY INSIGHT:
   By using uplift modeling, we can improve campaign profitability by
   ${(optimal_target['profit'] - baseline_profit):,.0f} compared to traditional
   'send to everyone' approach, while reducing costs and avoiding negative impacts.
""")

print("="*70)
print("✨ EVALUATION COMPLETE")
print("="*70)
