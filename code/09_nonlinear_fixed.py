import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('<PROJECT_ROOT>/modelB_win72_full.csv')
df['log_ctn_baseline'] = df['ctn_baseline'].apply(lambda x: np.log(x) if x > 0 else 0)

# Create features
df['traj_rising'] = (df['trajectory'] == 'Rising').astype(int)
df['traj_falling'] = (df['trajectory'] == 'Falling').astype(int)

# Non-linear features
df['delta_gt50'] = (df['delta_pct'] >= 50).astype(int)
df['delta_gt100'] = (df['delta_pct'] >= 100).astype(int)
df['delta_gt200'] = (df['delta_pct'] >= 200).astype(int)
df['peak_gt05'] = (df['max_val'] > 0.5).astype(int)
df['peak_gt10'] = (df['max_val'] > 1.0).astype(int)
df['peak_gt20'] = (df['max_val'] > 2.0).astype(int)
df['baseline_gt05'] = (df['ctn_baseline'] > 0.5).astype(int)

# Interactions
df['trajXsofa'] = df['traj_rising'] * df['sofa']
df['trajXage'] = df['traj_rising'] * df['age']
df['trajXctn'] = df['traj_rising'] * df['log_ctn_baseline']
df['fallXsofa'] = df['traj_falling'] * df['sofa']
df['fallXage'] = df['traj_falling'] * df['age']
df['fallXctn'] = df['traj_falling'] * df['log_ctn_baseline']

# Subgroups
df['sofa_high'] = (df['sofa'] > 8).astype(int)
df['sofa_low'] = (df['sofa'] <= 4).astype(int)

print(f"Dataset: {df.shape}")
print(f"Death rate: {df['death_28d'].mean():.3f}")
print()

# Feature sets
baseline_vars = ['age', 'sex', 'sofa', 'log_ctn_baseline']
linear_vars = baseline_vars + ['delta_pct']
nonlinear_vars = baseline_vars + ['delta_gt50', 'delta_gt100', 'delta_gt200', 'peak_gt05', 'peak_gt10', 'peak_gt20']
interaction_vars = baseline_vars + ['traj_rising', 'traj_falling', 'trajXsofa', 'trajXage', 'trajXctn', 'fallXsofa', 'fallXage', 'fallXctn']
all_vars = baseline_vars + ['delta_pct', 'traj_rising', 'traj_falling',
                            'delta_gt50', 'delta_gt100', 'delta_gt200', 'peak_gt05', 'peak_gt10', 'peak_gt20',
                            'trajXsofa', 'trajXage', 'trajXctn', 'fallXsofa', 'fallXage', 'fallXctn']

# Handle missing
for v in all_vars:
    df[v] = df[v].fillna(df[v].median())

def evaluate_model(X_train, y_train, X_test, y_test, model_name):
    model = LogisticRegression(max_iter=1000, random_state=20260101)
    model.fit(X_train, y_train)
    pred_train = model.predict_proba(X_train)[:, 1]
    pred_test = model.predict_proba(X_test)[:, 1]

    fpr_train, tpr_train, _ = roc_curve(y_train, pred_train)
    fpr_test, tpr_test, _ = roc_curve(y_test, pred_test)
    auc_train = auc(fpr_train, tpr_train)
    auc_test = auc(fpr_test, tpr_test)

    return {'Model': model_name, 'AUC_train': auc_train, 'AUC_test': auc_test}

# Split data
np.random.seed(20260101)
train_idx = np.random.choice(len(df), size=int(0.7 * len(df)), replace=False)
test_idx = np.setdiff1d(np.arange(len(df)), train_idx)

results = []

# Baseline
X_train = df.iloc[train_idx][baseline_vars].values
y_train = df.iloc[train_idx]['death_28d'].values
X_test = df.iloc[test_idx][baseline_vars].values
y_test = df.iloc[test_idx]['death_28d'].values
results.append(evaluate_model(X_train, y_train, X_test, y_test, 'Baseline'))

# + Linear Δ%cTn
X_train = df.iloc[train_idx][linear_vars].values
X_test = df.iloc[test_idx][linear_vars].values
results.append(evaluate_model(X_train, y_train, X_test, y_test, '+ Linear Δ%cTn'))

# + Non-linear features
X_train = df.iloc[train_idx][nonlinear_vars].values
X_test = df.iloc[test_idx][nonlinear_vars].values
results.append(evaluate_model(X_train, y_train, X_test, y_test, '+ Non-linear features'))

# + Interactions
X_train = df.iloc[train_idx][interaction_vars].values
X_test = df.iloc[test_idx][interaction_vars].values
results.append(evaluate_model(X_train, y_train, X_test, y_test, '+ Interactions'))

# All features
X_train = df.iloc[train_idx][all_vars].values
X_test = df.iloc[test_idx][all_vars].values
results.append(evaluate_model(X_train, y_train, X_test, y_test, 'All features'))

results_df = pd.DataFrame(results)
results_df['Delta_AUC_vs_Baseline'] = results_df['AUC_test'] - results_df.loc[0, 'AUC_test']
print("Model Comparison (All patients):")
print(results_df.round(3))
print()

# Save results
results_df.to_csv('<PROJECT_ROOT>/modelB_nonlinear_comparison.csv', index=False)
print("Saved: modelB_nonlinear_comparison.csv")
print()
