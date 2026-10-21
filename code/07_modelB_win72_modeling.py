import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('<PROJECT_ROOT>/modelB_win72_full.csv')
print(f"Dataset: {df.shape}")
print(f"Death rate: {df['death_28d'].mean():.3f}")
print()

# Feature sets
baseline_vars = ['age', 'sex', 'sofa', 'log_ctn_baseline']
delta_vars = baseline_vars + ['delta_pct']
traj_vars = baseline_vars + ['traj_rising', 'traj_falling']
all_vars = baseline_vars + ['delta_pct', 'traj_rising', 'traj_falling']

# Handle missing
for v in all_vars:
    df[v] = df[v].fillna(df[v].median())

# Split data
np.random.seed(20260101)
train_idx = np.random.choice(len(df), size=int(0.7 * len(df)), replace=False)
test_idx = np.setdiff1d(np.arange(len(df)), train_idx)

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

results = []

# Baseline
X_train = df.iloc[train_idx][baseline_vars].values
y_train = df.iloc[train_idx]['death_28d'].values
X_test = df.iloc[test_idx][baseline_vars].values
y_test = df.iloc[test_idx]['death_28d'].values
results.append(evaluate_model(X_train, y_train, X_test, y_test, 'Baseline'))

# + Delta %cTn
X_train = df.iloc[train_idx][delta_vars].values
X_test = df.iloc[test_idx][delta_vars].values
results.append(evaluate_model(X_train, y_train, X_test, y_test, '+ Delta %cTn'))

# + Trajectory dummies
X_train = df.iloc[train_idx][traj_vars].values
X_test = df.iloc[test_idx][traj_vars].values
results.append(evaluate_model(X_train, y_train, X_test, y_test, '+ Trajectory dummies'))

# All features
X_train = df.iloc[train_idx][all_vars].values
X_test = df.iloc[test_idx][all_vars].values
results.append(evaluate_model(X_train, y_train, X_test, y_test, 'All features'))

results_df = pd.DataFrame(results)
results_df['Delta_AUC_vs_Baseline'] = results_df['AUC_test'] - results_df.loc[0, 'AUC_test']
print("Model Comparison:")
print(results_df.round(3))
print()

results_df.to_csv('<PROJECT_ROOT>/modelB_win72_comparison.csv', index=False)
print("Saved: modelB_win72_comparison.csv")
print()

# ========== XGBoost ==========
print("=== XGBoost ===")
try:
    import xgboost as xgb

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 4,
        'eta': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 10,
        'seed': 20260101
    }

    xgb_model = xgb.train(params, dtrain, num_boost_round=300,
                          evals=[(dtrain, 'train'), (dtest, 'test')],
                          early_stopping_rounds=30, verbose_eval=False)

    pred_xgb_train = xgb_model.predict(dtrain)
    pred_xgb_test = xgb_model.predict(dtest)

    fpr_train, tpr_train, _ = roc_curve(y_train, pred_xgb_train)
    fpr_test, tpr_test, _ = roc_curve(y_test, pred_xgb_test)
    auc_xgb_train = auc(fpr_train, tpr_train)
    auc_xgb_test = auc(fpr_test, tpr_test)

    print(f"XGBoost AUC train: {auc_xgb_train:.3f}, test: {auc_xgb_test:.3f}")
    print()

    importance = xgb_model.get_score(importance_type='gain')
    importance_df = pd.DataFrame({
        'Feature': list(importance.keys()),
        'Importance': list(importance.values())
    }).sort_values('Importance', ascending=False)

    print("Top 10 important features:")
    print(importance_df.head(10))
    print()

    xgb_results = pd.DataFrame({
        'Model': ['XGBoost (all features)'],
        'AUC_train': [auc_xgb_train],
        'AUC_test': [auc_xgb_test],
        'Delta_AUC_vs_Baseline': [auc_xgb_test - results_df.loc[0, 'AUC_test']]
    })
    xgb_results.to_csv('<PROJECT_ROOT>/modelB_win72_xgboost.csv', index=False)
    importance_df.to_csv('<PROJECT_ROOT>/modelB_win72_importance.csv', index=False)

    print("Saved: modelB_win72_xgboost.csv, modelB_win72_importance.csv")

except ImportError:
    print("XGBoost not available")

print()
print("=== Analysis Complete ===")
