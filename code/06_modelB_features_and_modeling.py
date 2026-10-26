import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# Load data
dat = pd.read_csv('<PROJECT_ROOT>/modelB_newdef.csv')
print(f"Dataset: {dat.shape}")
print(f"Death rate: {dat['death_28d'].mean():.3f}")

# ========== Feature Engineering ==========
dat['log_ctn_baseline'] = np.log(dat['ctn_baseline'])
dat['log_ctn_followup'] = np.log(dat['ctn_followup'])

# Trajectory dummy variables
dat['traj_rising'] = (dat['trajectory'] == 'Rising').astype(int)
dat['traj_falling'] = (dat['trajectory'] == 'Falling').astype(int)
dat['traj_stable'] = (dat['trajectory'] == 'Stable').astype(int)

# Rule-based posterior probabilities
def soft_prob_rising(delta_pct):
    return min(0.95, 0.5 + delta_pct / 400) if delta_pct >= 20 else max(0.05, 0.5 + delta_pct / 400)

def soft_prob_falling(delta_pct):
    return min(0.95, 0.5 - delta_pct / 400) if delta_pct <= -20 else max(0.05, 0.5 - delta_pct / 400)

def soft_prob_stable(delta_pct):
    return max(0.05, 1 - abs(delta_pct) / 50)

dat['prob_rising'] = dat['delta_pct'].apply(soft_prob_rising)
dat['prob_falling'] = dat['delta_pct'].apply(soft_prob_falling)
dat['prob_stable'] = dat['delta_pct'].apply(soft_prob_stable)

prob_sum = dat['prob_rising'] + dat['prob_falling'] + dat['prob_stable']
dat['prob_rising'] = dat['prob_rising'] / prob_sum
dat['prob_falling'] = dat['prob_falling'] / prob_sum
dat['prob_stable'] = dat['prob_stable'] / prob_sum

# ========== GMM for trajectory classes ==========
print("\n=== GMM Trajectory Classification ===")
gmm_features = ['ctn_baseline', 'delta_pct', 'time_gap']
X_gmm = dat[gmm_features].values

scaler = StandardScaler()
X_gmm_scaled = scaler.fit_transform(X_gmm)

gmm = GaussianMixture(n_components=3, random_state=20260101, max_iter=200)
gmm.fit(X_gmm_scaled)

gmm_probs = gmm.predict_proba(X_gmm_scaled)
dat['prob_gmm_0'] = gmm_probs[:, 0]
dat['prob_gmm_1'] = gmm_probs[:, 1]
dat['prob_gmm_2'] = gmm_probs[:, 2]

gmm_means = dat.groupby(gmm.predict(X_gmm_scaled))['delta_pct'].mean().sort_values()
gmm_labels = {gmm_means.index[0]: 'Falling', gmm_means.index[1]: 'Stable', gmm_means.index[2]: 'Rising'}
dat['class_gmm'] = gmm.predict(X_gmm_scaled)
dat['class_gmm_label'] = dat['class_gmm'].map(gmm_labels)

print("GMM class distribution:")
print(dat['class_gmm_label'].value_counts())
print("\nGMM class death rates:")
print(dat.groupby('class_gmm_label')['death_28d'].agg(['count', 'mean']).round(3))

# Save features
feature_cols = ['stay_id', 'age', 'sex', 'sofa', 'ctn_baseline', 'log_ctn_baseline',
                'ctn_followup', 'log_ctn_followup', 'delta_pct', 'time_gap',
                'traj_rising', 'traj_falling', 'traj_stable',
                'prob_rising', 'prob_falling', 'prob_stable',
                'prob_gmm_0', 'prob_gmm_1', 'prob_gmm_2',
                'class_gmm_label', 'death_28d']

features_df = dat[feature_cols].copy()
features_df.to_csv('<PROJECT_ROOT>/modelB_features.csv', index=False)
print(f"\nFeatures saved: modelB_features.csv ({features_df.shape})")


# ========== Model Comparison: Baseline vs +Trajectory vs +GMM ==========
print("\n=== Model Comparison ===")

baseline_vars = ['age', 'sex', 'sofa', 'log_ctn_baseline']
delta_vars = baseline_vars + ['delta_pct']
traj_vars = baseline_vars + ['traj_rising', 'traj_falling']
prob_vars = baseline_vars + ['prob_rising', 'prob_falling', 'prob_stable']
gmm_vars = baseline_vars + ['prob_gmm_0', 'prob_gmm_1', 'prob_gmm_2']
all_vars = baseline_vars + ['delta_pct', 'traj_rising', 'traj_falling',
                            'prob_rising', 'prob_falling', 'prob_stable',
                            'prob_gmm_0', 'prob_gmm_1', 'prob_gmm_2']

# Handle missing
for v in all_vars:
    if v in dat.columns:
        dat[v] = dat[v].fillna(dat[v].median())

# Split data
np.random.seed(20260101)
train_idx = np.random.choice(len(dat), size=int(0.7 * len(dat)), replace=False)
test_idx = np.setdiff1d(np.arange(len(dat)), train_idx)

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
X_train = dat.iloc[train_idx][baseline_vars].values
y_train = dat.iloc[train_idx]['death_28d'].values
X_test = dat.iloc[test_idx][baseline_vars].values
y_test = dat.iloc[test_idx]['death_28d'].values
results.append(evaluate_model(X_train, y_train, X_test, y_test, 'Baseline'))

# + Delta %cTn
X_train = dat.iloc[train_idx][delta_vars].values
X_test = dat.iloc[test_idx][delta_vars].values
results.append(evaluate_model(X_train, y_train, X_test, y_test, '+ Delta %cTn'))

# + Trajectory dummies
X_train = dat.iloc[train_idx][traj_vars].values
X_test = dat.iloc[test_idx][traj_vars].values
results.append(evaluate_model(X_train, y_train, X_test, y_test, '+ Trajectory dummies'))

# + Trajectory probabilities
X_train = dat.iloc[train_idx][prob_vars].values
X_test = dat.iloc[test_idx][prob_vars].values
results.append(evaluate_model(X_train, y_train, X_test, y_test, '+ Trajectory probs'))

# + GMM probabilities
X_train = dat.iloc[train_idx][gmm_vars].values
X_test = dat.iloc[test_idx][gmm_vars].values
results.append(evaluate_model(X_train, y_train, X_test, y_test, '+ GMM probs'))

# All features
X_train = dat.iloc[train_idx][all_vars].values
X_test = dat.iloc[test_idx][all_vars].values
results.append(evaluate_model(X_train, y_train, X_test, y_test, 'All features'))

results_df = pd.DataFrame(results)
results_df['Delta_AUC_vs_Baseline'] = results_df['AUC_test'] - results_df.loc[0, 'AUC_test']
print("\nModel Comparison:")
print(results_df.round(3))

results_df.to_csv('<PROJECT_ROOT>/modelB_model_comparison_newdef.csv', index=False)
print("\nSaved: modelB_model_comparison_newdef.csv")

# ========== XGBoost ==========
print("\n=== XGBoost ===")
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

    importance = xgb_model.get_score(importance_type='gain')
    importance_df = pd.DataFrame({
        'Feature': list(importance.keys()),
        'Importance': list(importance.values())
    }).sort_values('Importance', ascending=False)

    print("\nTop 10 important features:")
    print(importance_df.head(10))

    xgb_results = pd.DataFrame({
        'Model': ['XGBoost (all features)'],
        'AUC_train': [auc_xgb_train],
        'AUC_test': [auc_xgb_test],
        'Delta_AUC_vs_Baseline': [auc_xgb_test - results_df.loc[0, 'AUC_test']]
    })
    xgb_results.to_csv('<PROJECT_ROOT>/modelB_xgboost_results.csv', index=False)
    importance_df.to_csv('<PROJECT_ROOT>/modelB_xgboost_importance.csv', index=False)

    print("\nSaved: modelB_xgboost_results.csv, modelB_xgboost_importance.csv")

except ImportError:
    print("XGBoost not available, skipping...")

print("\n=== Analysis Complete ===")
