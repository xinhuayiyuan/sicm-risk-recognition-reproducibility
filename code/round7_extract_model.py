# 124_extract_primary_model.py — export the locked primary LASSO formula without changing analysis choices
# -*- coding: utf-8 -*-
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
SEED = 20260101
D = Path(__file__).resolve().parents[1] / 'exec'
co = pd.read_csv(D / 'data/round7_cohort_timing.csv')
co['aki_stage_24h'] = co['aki_stage_24h'].fillna(0).astype(int)
d = co[(co.has_hf == 0) & (co.congestive_heart_failure == 0)].copy().reset_index(drop=True)
NUM = ['age', 'cci', 'sofa_24h', 'heart_rate_max', 'mbp_min', 'resp_rate_max',
       'spo2_min', 'temperature_max', 'lactate_max_24h', 'wbc_max', 'platelets_min',
       'creatinine_max', 'bun_max', 'bilirubin_total_max', 'sodium_min', 'potassium_max',
       'glucose_max', 'inr_max', 'ptt_max', 'bicarbonate_min', 'aniongap_max',
       'hemoglobin_min', 'ne_max_24h_strict', 'aki_stage_24h']
COMORB = ['myocardial_infarct', 'peripheral_vascular_disease', 'cerebrovascular_disease',
          'dementia', 'chronic_pulmonary_disease', 'rheumatic_disease', 'mild_liver_disease',
          'diabetes_without_cc', 'diabetes_with_cc', 'renal_disease', 'malignant_cancer',
          'severe_liver_disease', 'metastatic_solid_tumor', 'aids']
BIN = ['gender_f', 'race_black', 'race_other'] + COMORB + [
    'vent_24h', 'rrt_firstday', 'lactate_miss', 'bili_miss']
d['gender_f'] = (d.gender == 'F').astype(int)
rc = d.race.astype(str).str.upper()
d['race_black'] = rc.str.contains('BLACK').astype(int)
d['race_other'] = (~rc.str.contains('WHITE|BLACK')).astype(int)
for c in COMORB + ['rrt_firstday']:
    d[c] = d[c].fillna(0).astype(int)
FEATS = [f for f in NUM + BIN if f != 'aids']
y = d.sicm.to_numpy()
tr, te = train_test_split(d.index, test_size=0.3, stratify=y, random_state=SEED)
med = d.loc[tr, NUM].median()
xtr = d.loc[tr, FEATS].copy()
xte = d.loc[te, FEATS].copy()
xtr[NUM] = xtr[NUM].fillna(med)
xte[NUM] = xte[NUM].fillna(med)
scaler = StandardScaler().fit(xtr[NUM])
xtr[NUM] = scaler.transform(xtr[NUM])
xte[NUM] = scaler.transform(xte[NUM])
model = LogisticRegressionCV(Cs=20, penalty='l1', solver='liblinear', cv=10,
                             scoring='roc_auc', max_iter=3000,
                             random_state=SEED).fit(xtr, y[tr])
p = model.predict_proba(xte)[:, 1]
auc = roc_auc_score(y[te], p)
coef = pd.Series(model.coef_[0], index=FEATS)
selected = coef[coef.abs() > 1e-8]
previous = pd.read_csv(D / 'results/round7_lasso_std_coefficients.csv').set_index('feature').std_coef
assert len(d) == 613 and len(tr) == 429 and len(te) == 184
assert int(y[tr].sum()) == 78 and int(y[te].sum()) == 33
assert len(selected) == 44 and round(auc, 3) == 0.634
assert set(selected.index) == set(previous.index)
comparison = pd.DataFrame({'exported': selected.loc[previous.index],
                           'previous': previous,
                           'difference': selected.loc[previous.index] - previous})
print(comparison.to_string())
assert np.allclose(selected.loc[previous.index], previous.values, atol=0.001, rtol=0)
rows = []
for feature in FEATS:
    numeric = feature in NUM
    rows.append({'feature': feature, 'feature_type': 'continuous' if numeric else 'binary',
                 'selected': int(abs(coef[feature]) > 1e-8),
                 'train_imputation_median': med[feature] if numeric else np.nan,
                 'train_mean': scaler.mean_[NUM.index(feature)] if numeric else np.nan,
                 'train_sd_population': scaler.scale_[NUM.index(feature)] if numeric else np.nan,
                 'penalised_coefficient': coef[feature]})
rows.append({'feature': '(Intercept)', 'feature_type': 'intercept', 'selected': 1,
             'train_imputation_median': np.nan, 'train_mean': np.nan,
             'train_sd_population': np.nan, 'penalised_coefficient': model.intercept_[0]})
out = pd.DataFrame(rows)
out.to_csv(D / 'results/round7_primary_lasso_model_parameters.csv', index=False)
artifact = {
    'model': 'L1-penalised logistic regression; LogisticRegressionCV lambda.min equivalent',
    'formula': 'logit(p) = intercept + sum(coefficient[j] * x[j]); continuous x=(value-mean)/sd_population after median imputation; binary x is unscaled',
    'development_n': len(tr), 'development_events': int(y[tr].sum()),
    'test_n': len(te), 'test_events': int(y[te].sum()), 'seed': SEED,
    'solver': 'liblinear', 'penalty': 'l1', 'cv_folds': 10, 'cs': 20,
    'scoring': 'roc_auc', 'max_iter': 3000, 'selected_C': float(model.C_[0]),
    'intercept': float(model.intercept_[0]), 'test_auc': float(auc),
    'features': FEATS, 'numeric_features': NUM,
    'coefficients': {k: float(v) for k, v in coef.items()},
    'preprocessing': {f: {'imputation_median': float(med[f]),
                          'mean': float(scaler.mean_[i]),
                          'sd_population': float(scaler.scale_[i])}
                      for i, f in enumerate(NUM)}
}
(D / 'results/round7_primary_lasso_model.json').write_text(
    json.dumps(artifact, indent=2, ensure_ascii=False), encoding='utf-8')
print(f'Exported {len(selected)} non-zero coefficients + intercept; development={len(tr)}/{int(y[tr].sum())} events; test AUC={auc:.6f}; C={model.C_[0]:.8f}')
