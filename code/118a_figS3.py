# 118a_figS3.py — figS3: 床旁评分ROC + 修正界标ROC
# -*- coding: utf-8 -*-
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve

SEED = 20260101
D = '<PROJECT_ROOT>/exec'

co = pd.read_csv(f'{D}/data/cohort_main.csv')
aki = pd.read_csv(f'{D}/data/aki_stage_24h.csv')
st = pd.read_csv(f'{D}/data/cohort_dn_strict24h.csv',
                 usecols=['stay_id', 'ne_max_24h_strict', 'dobutamine_24h',
                          'milrinone_24h', 'vent_24h'])
co = co.merge(aki, on='stay_id', how='left').merge(st, on='stay_id', how='left')
co['aki_stage_24h'] = co['aki_stage_24h'].fillna(0).astype(int)
dn = co[(co['has_hf'] == 0) & (co['congestive_heart_failure'] == 0)].copy().reset_index(drop=True)

NUM = ['age','cci','sofa_24h','heart_rate_max','mbp_min','resp_rate_max','spo2_min',
       'temperature_max','lactate_max_24h','wbc_max','platelets_min','creatinine_max',
       'bun_max','bilirubin_total_max','sodium_min','potassium_max','glucose_max',
       'inr_max','ptt_max','bicarbonate_min','aniongap_max','hemoglobin_min',
       'ne_max_24h_strict','aki_stage_24h']
COMORB = ['myocardial_infarct','peripheral_vascular_disease','cerebrovascular_disease',
          'dementia','chronic_pulmonary_disease','rheumatic_disease','mild_liver_disease',
          'diabetes_without_cc','diabetes_with_cc','renal_disease','malignant_cancer',
          'severe_liver_disease','metastatic_solid_tumor','aids']
BIN = ['gender_f','race_black','race_other'] + COMORB + \
      ['vent_24h','dobutamine_24h','milrinone_24h','rrt_firstday','lactate_miss','bili_miss']
dn['gender_f'] = (dn['gender'] == 'F').astype(int)
rc = dn['race'].astype(str).str.upper()
dn['race_black'] = rc.str.contains('BLACK').astype(int)
dn['race_other'] = (~rc.str.contains('WHITE|BLACK')).astype(int)
for c in COMORB + ['rrt_firstday']:
    dn[c] = dn[c].fillna(0).astype(int)
FEATS = [f for f in (NUM + BIN) if f != 'aids']

def prep(a, b):
    med = a[NUM].median()
    Xa = a[FEATS].copy(); Xb = b[FEATS].copy()
    Xa[NUM] = Xa[NUM].fillna(med); Xb[NUM] = Xb[NUM].fillna(med)
    sc = StandardScaler().fit(Xa[NUM])
    Xa[NUM] = sc.transform(Xa[NUM]); Xb[NUM] = sc.transform(Xb[NUM])
    return Xa, Xb

def las_fit20(X, yy):
    return LogisticRegressionCV(Cs=20, penalty='l1', solver='liblinear', cv=10,
                                scoring='roc_auc', max_iter=3000, random_state=SEED).fit(X, yy)

y = dn['sicm'].values
tr, te = train_test_split(dn.index, test_size=0.3, stratify=y, random_state=SEED)
dtr, dte = dn.loc[tr].copy(), dn.loc[te].copy()
Xtr, Xte = prep(dtr, dte)
ytr, yte = y[tr], y[te]
las = las_fit20(Xtr, ytr)
p1 = las.predict_proba(Xte)[:, 1]

# ---- 床旁评分（99e①流程） ----
sc_df = pd.read_csv(f'{D}/results/v5_score_card_full.csv')
sel = [f for f in sc_df.feature if f != '(Intercept)']
pts_raw = sc_df.set_index('feature').loc[sel, 'points_raw']
cont = [v for v in sel if v in NUM]
mu = dtr[cont].mean(); sd_ = dtr[cont].std().replace(0, 1)
dte_z = dte[sel].astype(float).copy()
dte_z[cont] = (dte_z[cont].fillna(dtr[cont].median()) - mu) / sd_
tot_te = (dte_z * pts_raw).sum(axis=1)
auc_score = roc_auc_score(yte, tot_te)
print(f'[figS3] bedside score test AUC={auc_score:.3f} (manuscript: 0.719)')

# ---- 修正界标（106流程） ----
cr = pd.read_csv(f'{D}/data/comprisk_corrected.csv')[['stay_id', 't_ef50', 't_death']]
dn_lm = dn.merge(cr, on='stay_id', how='left')
early = (dn_lm['sicm'] == 1) & (dn_lm['t_ef50'] <= 24)
decline_only = (dn_lm['sicm'] == 1) & (dn_lm['t_ef50'].isna())
dead24 = dn_lm['t_death'].fillna(np.inf) <= 24
lm = dn_lm[~early & ~decline_only & ~dead24].copy().reset_index(drop=True)
ylm = ((lm['sicm'] == 1) & (lm['t_ef50'] > 24)).astype(int).values
tr2, te2 = train_test_split(np.arange(len(lm)), test_size=0.3, stratify=ylm, random_state=SEED)
Xa, Xb = prep(lm.iloc[tr2], lm.iloc[te2])
p_trans = las.predict_proba(Xb)[:, 1]
m_lm = las_fit20(Xa, ylm[tr2])
p_lm = m_lm.predict_proba(Xb)[:, 1]
print(f'[figS3] landmark n={len(lm)}, transport AUC={roc_auc_score(ylm[te2], p_trans):.3f}, '
      f'refit AUC={roc_auc_score(ylm[te2], p_lm):.3f}')

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
ax = axes[0]
for name, pp, c in [('Full LASSO model (AUC 0.731)', p1, '#d62728'),
                    (f'Integer bedside score (AUC {auc_score:.3f})', tot_te, '#1f77b4')]:
    fpr, tpr, _ = roc_curve(yte, pp)
    ax.plot(fpr, tpr, color=c, lw=1.6, label=name)
ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate')
ax.set_title('A. New-onset cohort: held-out test set\n(n = 237; 65 SICM)', fontsize=11)
ax.legend(fontsize=9, loc='lower right'); ax.grid(alpha=0.3)
ax = axes[1]
for name, pp, c in [(f'Frozen model transported (AUC {roc_auc_score(ylm[te2], p_trans):.3f})', p_trans, '#d62728'),
                    (f'Within-landmark refit (AUC {roc_auc_score(ylm[te2], p_lm):.3f})', p_lm, '#7f7f7f')]:
    fpr, tpr, _ = roc_curve(ylm[te2], pp)
    ax.plot(fpr, tpr, color=c, lw=1.6, label=name)
ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate')
ax.set_title('B. Landmark subset of new-onset cohort\n(n = 610; 60 SICM after 24 h)', fontsize=11)
ax.legend(fontsize=9, loc='lower right'); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f'{D}/figures/figS3_score_landmark_roc.png', dpi=200)
plt.close('all'); print('[done 118a: figS3 saved]')
