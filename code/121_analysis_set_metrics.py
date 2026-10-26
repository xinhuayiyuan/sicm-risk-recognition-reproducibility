# 121_analysis_set_metrics.py — analysis-set checks without changing primary estimates
# -*- coding: utf-8 -*-
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

SEED = 20260101
D = '<PROJECT_ROOT>/exec'
co = pd.read_csv(f'{D}/data/cohort_main.csv')
aki = pd.read_csv(f'{D}/data/aki_stage_24h.csv')
st = pd.read_csv(f'{D}/data/cohort_dn_strict24h.csv', usecols=['stay_id', 'ne_max_24h_strict', 'vent_24h'])
co = co.merge(aki, on='stay_id', how='left').merge(st, on='stay_id', how='left')
co['aki_stage_24h'] = co['aki_stage_24h'].fillna(0).astype(int)
d = co[(co.has_hf == 0) & (co.congestive_heart_failure == 0)].copy().reset_index(drop=True)
NUM = ['age','cci','sofa_24h','heart_rate_max','mbp_min','resp_rate_max','spo2_min','temperature_max','lactate_max_24h','wbc_max','platelets_min','creatinine_max','bun_max','bilirubin_total_max','sodium_min','potassium_max','glucose_max','inr_max','ptt_max','bicarbonate_min','aniongap_max','hemoglobin_min','ne_max_24h_strict','aki_stage_24h']
COMORB = ['myocardial_infarct','peripheral_vascular_disease','cerebrovascular_disease','dementia','chronic_pulmonary_disease','rheumatic_disease','mild_liver_disease','diabetes_without_cc','diabetes_with_cc','renal_disease','malignant_cancer','severe_liver_disease','metastatic_solid_tumor','aids']
BIN = ['gender_f','race_black','race_other'] + COMORB + ['vent_24h','rrt_firstday','lactate_miss','bili_miss']
d['gender_f'] = (d.gender == 'F').astype(int)
rc = d.race.astype(str).str.upper()
d['race_black'] = rc.str.contains('BLACK').astype(int)
d['race_other'] = (~rc.str.contains('WHITE|BLACK')).astype(int)
for c in COMORB + ['rrt_firstday']:
    d[c] = d[c].fillna(0).astype(int)
FEATS = [f for f in NUM + BIN if f != 'aids']
y = d.sicm.to_numpy()
def prep(a, b):
    med = a[NUM].median(); xa = a[FEATS].copy(); xb = b[FEATS].copy()
    xa[NUM] = xa[NUM].fillna(med); xb[NUM] = xb[NUM].fillna(med)
    sc = StandardScaler().fit(xa[NUM]); xa[NUM] = sc.transform(xa[NUM]); xb[NUM] = sc.transform(xb[NUM])
    return xa, xb
def fit(x, yy):
    return LogisticRegressionCV(Cs=20, penalty='l1', solver='liblinear', cv=10, scoring='roc_auc', max_iter=3000, random_state=SEED).fit(x, yy)
def boot_ci(yy, pp, seed=20261026, n=2000):
    rr = np.random.RandomState(seed); vals = []
    for _ in range(n):
        pos = rr.choice(np.where(yy == 1)[0], size=(yy == 1).sum(), replace=True)
        neg = rr.choice(np.where(yy == 0)[0], size=(yy == 0).sum(), replace=True)
        ii = np.r_[pos, neg]; vals.append(roc_auc_score(yy[ii], pp[ii]))
    return np.percentile(vals, [2.5, 97.5])
tr, te = train_test_split(d.index, test_size=0.3, stratify=y, random_state=SEED)
xtr, xte = prep(d.loc[tr], d.loc[te]); m = fit(xtr, y[tr]); p = m.predict_proba(xte)[:, 1]; yt = y[te]
score = pd.read_csv(f'{D}/results/v5_score_card_full.csv')
sel = [f for f in score.feature if f != '(Intercept)']; pts = score.set_index('feature').loc[sel, 'points_raw']
cont = [f for f in sel if f in NUM]; mu = d.loc[tr, cont].mean(); sd = d.loc[tr, cont].std().replace(0, 1)
z = d.loc[te, sel].astype(float).copy(); z[cont] = (z[cont].fillna(d.loc[tr, cont].median()) - mu) / sd
sp = (z * pts).sum(axis=1).to_numpy(); sci = boot_ci(yt, sp)
oof = np.zeros(len(d))
folds = []
for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(d, y), 1):
    xa, xb = prep(d.iloc[a], d.iloc[b]); oof[b] = fit(xa, y[a]).predict_proba(xb)[:, 1]
    folds.append({'fold': k, 'n': len(b), 'events': int(y[b].sum()), 'AUC': roc_auc_score(y[b], oof[b])})
oci = boot_ci(y, oof)
fpr, tpr, thresholds = roc_curve(yt, p)
valid = np.isfinite(thresholds) & (thresholds <= 1)
idx = np.where(valid)[0][np.argmin(np.abs(tpr[valid] - 0.65))]
th = thresholds[idx]; pred = p >= th; tn, fp, fn, tp = confusion_matrix(yt, pred).ravel()
metrics = [
    ('Descriptive cohort', len(co), int(co.sicm.sum()), np.nan, np.nan, np.nan, np.nan, np.nan),
    ('New-onset cohort', len(d), int(y.sum()), np.nan, np.nan, np.nan, np.nan, np.nan),
    ('Held-out test: primary model', len(yt), int(yt.sum()), roc_auc_score(yt, p), np.nan, np.nan, np.nan, np.nan),
    ('Held-out test: integer bedside score', len(yt), int(yt.sum()), roc_auc_score(yt, sp), sci[0], sci[1], np.nan, np.nan),
    ('Five-fold OOF: primary specification', len(y), int(y.sum()), roc_auc_score(y, oof), oci[0], oci[1], np.nan, np.nan),
    ('Test threshold nearest sensitivity 0.65 (exploratory)', len(yt), int(yt.sum()), np.nan, np.nan, np.nan, tp/(tp+fn), tn/(tn+fp)),
]
out = pd.DataFrame(metrics, columns=['analysis_set','n','events','AUC','AUC_lo','AUC_hi','sensitivity','specificity'])
out['threshold'] = np.nan; out.loc[out.index[-1], 'threshold'] = th; out['flagged_n'] = np.nan; out.loc[out.index[-1], 'flagged_n'] = int(pred.sum())
out.to_csv(f'{D}/results/v7_analysis_set_metrics.csv', index=False)
pd.DataFrame(folds).to_csv(f'{D}/results/v7_oof_fold_details.csv', index=False)
print(out.to_string(index=False)); print(pd.DataFrame(folds).to_string(index=False))
