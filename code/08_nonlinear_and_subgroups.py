import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('<PROJECT_ROOT>/modelB_win72_full.csv')
df['log_ctn_baseline'] = df['ctn_baseline'].apply(lambda x: np.log(x) if x > 0 else 0)

print(f"Dataset: {df.shape}")
print()

# ========== 1. Non-linear Trajectory Features ==========
print("=== Non-linear Trajectory Features ===")
print()

# Create non-linear features
df['delta_gt50'] = (df['delta_pct'] >= 50).astype(int)
df['delta_gt100'] = (df['delta_pct'] >= 100).astype(int)
df['delta_gt200'] = (df['delta_pct'] >= 200).astype(int)
df['peak_gt05'] = (df['max_val'] > 0.5).astype(int)
df['peak_gt10'] = (df['max_val'] > 1.0).astype(int)
df['peak_gt20'] = (df['max_val'] > 2.0).astype(int)
df['baseline_gt05'] = (df['ctn_baseline'] > 0.5).astype(int)
df['baseline_gt10'] = (df['ctn_baseline'] > 1.0).astype(int)

features = ['delta_gt50', 'delta_gt100', 'delta_gt200', 'peak_gt05', 'peak_gt10', 'peak_gt20', 'baseline_gt05', 'baseline_gt10']

print("Feature prevalence:")
for f in features:
    print(f"  {f}: {df[f].mean():.1%} (n={df[f].sum()})")

print()
print("Death rate by feature:")
for f in features:
    if df[f].sum() > 0:
        dr = df[df[f]==1]['death_28d'].mean()
        n = df[f].sum()
        print(f"  {f}: {dr:.1%} (n={n})")

print()

# ========== 2. Subgroup Analysis ==========
print("=== Subgroup Analysis ===")
print()

# Create subgroups
df['sofa_high'] = (df['sofa'] > 8).astype(int)
df['sofa_low'] = (df['sofa'] <= 4).astype(int)
df['age_high'] = (df['age'] > 75).astype(int)
df['age_low'] = (df['age'] <= 65).astype(int)
df['ctn_high'] = (df['ctn_baseline'] > 0.5).astype(int)
df['ctn_low'] = (df['ctn_baseline'] <= 0.1).astype(int)

subgroups = ['sofa_high', 'sofa_low', 'age_high', 'age_low', 'ctn_high', 'ctn_low']

print("Subgroup sizes and death rates:")
for sg in subgroups:
    n = df[sg].sum()
    dr = df[df[sg]==1]['death_28d'].mean()
    dr0 = df[df[sg]==0]['death_28d'].mean()
    print(f"  {sg}: n={n}, death={dr:.1%} vs {dr0:.1%} (diff={dr-dr0:.1%})")

print()
print("Trajectory-death rate by subgroup:")
for sg in subgroups:
    print(f"  {sg}:")
    sub = df[df[sg]==1]
    if len(sub) > 0:
        print(sub.groupby('trajectory')['death_28d'].agg(['count','mean']).round(3))
    print()

# ========== 3. Interaction Analysis ==========
print("=== Interaction Analysis ===")
print()

# Create trajectory dummies
df['traj_rising'] = (df['trajectory'] == 'Rising').astype(int)
df['traj_falling'] = (df['trajectory'] == 'Falling').astype(int)

# Create interaction terms
df['trajXsofa'] = df['traj_rising'] * df['sofa']
df['trajXage'] = df['traj_rising'] * df['age']
df['trajXctn'] = df['traj_rising'] * df['log_ctn_baseline']
df['fallXsofa'] = df['traj_falling'] * df['sofa']
df['fallXage'] = df['traj_falling'] * df['age']
df['fallXctn'] = df['traj_falling'] * df['log_ctn_baseline']

interactions = ['trajXsofa', 'trajXage', 'trajXctn', 'fallXsofa', 'fallXage', 'fallXctn']

print("Interaction terms created:")
for i in interactions:
    print(f"  {i}: mean={df[i].mean():.2f}")

print()
print("Death rate by interaction (high vs low median):")
for i in interactions:
    if df[i].std() > 0:
        med = df[i].median()
        dr = df[df[i]>med]['death_28d'].mean()
        dr0 = df[df[i]<=med]['death_28d'].mean()
        print(f"  {i}: high={dr:.1%} vs low={dr0:.1%} (diff={dr-dr0:.1%})")

print()
print("=== Analysis Complete ===")
