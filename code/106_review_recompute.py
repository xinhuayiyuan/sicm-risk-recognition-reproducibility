# 106_review_recompute.py — landmark-24 h cohort derivation, COVID-era
# exclusion and decline-criterion exclusion sensitivity analyses.
# Same pipeline as 101_full_recompute_strict24h.py (same seed, features, fits).
# -*- coding: utf-8 -*-
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

SEED = 20260101
rng = np.random.RandomState(SEED)
D = 'D:/MIMICIV/exec'
OUT = []

def log(s):
    print(s); OUT.append(str(s))

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

def enrich(d):
    d = d.copy()
    d['gender_f'] = (d['gender'] == 'F').astype(int)
    rc = d['race'].astype(str).str.upper()
    d['race_black'] = rc.str.contains('BLACK').astype(int)
    d['race_other'] = (~rc.str.contains('WHITE|BLACK')).astype(int)
    for c in COMORB + ['rrt_firstday']:
        d[c] = d[c].fillna(0).astype(int)
    return d

dn = enrich(dn)
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

def boot_ci(yy, pp, n=1000):
    a = []
    for _ in range(n):
        i = rng.randint(0, len(yy), len(yy))
        if len(np.unique(yy[i])) > 1:
            a.append(roc_auc_score(yy[i], pp[i]))
    return np.percentile(a, [2.5, 97.5])

CONF = ['age', 'sofa_24h', 'cci', 'lact_f', 'vent_24h', 'aki_stage_24h']
dn['lact_f'] = dn['lactate_max_24h'].fillna(dn['lactate_max_24h'].median())
dn['vent_24h'] = dn['vent_24h'].fillna(0).astype(int)

def adj_or(dsub, xcol):
    X = sm.add_constant(pd.concat([dsub[xcol], dsub[CONF]], axis=1).astype(float))
    m = sm.Logit(dsub['mort_28d'].astype(float), X).fit(disp=0)
    ci = np.exp(m.conf_int().loc[xcol])
    return np.exp(m.params[xcol]), ci[0], ci[1], m.pvalues[xcol]


# ---- Part 1: 冻结主模型（与101一致）+ 完整性校验 ----
y = dn['sicm'].values
tr, te = train_test_split(dn.index, test_size=0.3, stratify=y, random_state=SEED)
dtr, dte = dn.loc[tr].copy(), dn.loc[te].copy()
Xtr, Xte = prep(dtr, dte)
ytr, yte = y[tr], y[te]
las = las_fit20(Xtr, ytr)
p1 = las.predict_proba(Xte)[:, 1]
log(f'[sanity] frozen model test AUC={roc_auc_score(yte, p1):.3f} (manuscript: 0.731); test n={len(te)}, events={yte.sum()}')

# ---- Part 2 (A1): 修正landmark-24h ----
cr = pd.read_csv(f'{D}/data/comprisk_corrected.csv')[['stay_id', 't_ef50', 't_death']]
dn_lm = dn.merge(cr, on='stay_id', how='left')

lm_old = dn_lm[(dn_lm['t_ef50'].isna()) | (dn_lm['t_ef50'] > 24)]
ylm_old = ((lm_old['sicm'] == 1) & (lm_old['t_ef50'] > 24)).astype(int).values
log(f'\n[A1-old] n={len(lm_old)}, events={ylm_old.sum()} (v5 csv: 644/60)')

early = (dn_lm['sicm'] == 1) & (dn_lm['t_ef50'] <= 24)
decline_only = (dn_lm['sicm'] == 1) & (dn_lm['t_ef50'].isna())
dead24 = dn_lm['t_death'].fillna(np.inf) <= 24
log(f'[A1] excluded: early-onset(<=24h)={early.sum()}, decline-only(no t_ef50)={decline_only.sum()}, '
    f'death<=24h among rest={(dead24 & ~early & ~decline_only).sum()}')
lm = dn_lm[~early & ~decline_only & ~dead24].copy().reset_index(drop=True)
ylm = ((lm['sicm'] == 1) & (lm['t_ef50'] > 24)).astype(int).values
log(f'[A1-corrected] landmark n={len(lm)}, events after 24h={ylm.sum()} ({ylm.mean()*100:.1f}%), non-events={len(lm)-ylm.sum()}')
tr2, te2 = train_test_split(np.arange(len(lm)), test_size=0.3, stratify=ylm, random_state=SEED)
Xa, Xb = prep(lm.iloc[tr2], lm.iloc[te2])
p_trans = las.predict_proba(Xb)[:, 1]
auc_trans = roc_auc_score(ylm[te2], p_trans); ci_trans = boot_ci(ylm[te2], p_trans)
m_lm = las_fit20(Xa, ylm[tr2])
p_lm = m_lm.predict_proba(Xb)[:, 1]
auc_lm = roc_auc_score(ylm[te2], p_lm); ci_lm = boot_ci(ylm[te2], p_lm)
log(f'[A1-corrected] transport(frozen): AUC={auc_trans:.3f} ({ci_trans[0]:.3f}-{ci_trans[1]:.3f})')
log(f'[A1-corrected] within-landmark refit: AUC={auc_lm:.3f} ({ci_lm[0]:.3f}-{ci_lm[1]:.3f})')
pd.DataFrame({'analysis': ['landmark_cohort_n', 'events_after_24h', 'event_rate',
                           'excluded_early_onset', 'excluded_decline_only', 'excluded_death_le24h',
                           'transport_AUC', 'transport_lo', 'transport_hi',
                           'refit_AUC', 'refit_lo', 'refit_hi'],
              'value': [len(lm), int(ylm.sum()), round(ylm.mean(), 3),
                        int(early.sum()), int(decline_only.sum()), int((dead24 & ~early & ~decline_only).sum()),
                        round(auc_trans, 3), round(ci_trans[0], 3), round(ci_trans[1], 3),
                        round(auc_lm, 3), round(ci_lm[0], 3), round(ci_lm[1], 3)]}
             ).to_csv(f'{D}/results/v6_landmark24h_corrected.csv', index=False)

# ---- Part 3 (B4): COVID时代 ----
log('\n[B4] era composition of 789-cohort:')
t = dn.groupby('anchor_year_group')['sicm'].agg(['count', 'sum'])
log(t.to_string())
covid = dn['anchor_year_group'] == '2020 - 2022'
log(f'[B4] 2020-2022 block: n={covid.sum()} ({covid.mean()*100:.1f}%), events={dn.loc[covid,"sicm"].sum()}')
o, lo_, hi, pv = adj_or(dn, 'sicm')
log(f'[B4] full-cohort adjusted OR={o:.3f} ({lo_:.3f}-{hi:.3f}), P={pv:.4f}')
o2, lo2, hi2, pv2 = adj_or(dn[~covid], 'sicm')
log(f'[B4] excluding 2020-2022: n={(~covid).sum()}, events={dn.loc[~covid,"sicm"].sum()}, adjusted OR={o2:.3f} ({lo2:.3f}-{hi2:.3f}), P={pv2:.4f}')
pd.DataFrame({'analysis': ['covid_block_n', 'covid_block_events', 'covid_block_pct',
                           'adj_OR_full', 'adj_OR_full_lo', 'adj_OR_full_hi',
                           'adj_OR_excl_covid', 'adj_OR_excl_covid_lo', 'adj_OR_excl_covid_hi', 'adj_OR_excl_covid_p'],
              'value': [int(covid.sum()), int(dn.loc[covid, 'sicm'].sum()), round(covid.mean() * 100, 1),
                        round(o, 3), round(lo_, 3), round(hi, 3),
                        round(o2, 3), round(lo2, 3), round(hi2, 3), round(pv2, 4)]}
             ).to_csv(f'{D}/results/v6_covid_era_sensitivity.csv', index=False)

# ---- Part 4 (B10): 剔除13例decline-only ----
dec = (dn['sicm'] == 1) & dn_lm['t_ef50'].isna().values
log(f'\n[B10] decline-only cases excluded: {dec.sum()} (manuscript: 13)')
dn10 = dn[~dec].copy().reset_index(drop=True)
o3, lo3, hi3, pv3 = adj_or(dn10, 'sicm')
log(f'[B10] adjusted OR excl. decline-only (n={len(dn10)}, events={dn10.sicm.sum()}): OR={o3:.3f} ({lo3:.3f}-{hi3:.3f}), P={pv3:.4f}')
y10 = dn10['sicm'].values
tr10, te10 = train_test_split(dn10.index, test_size=0.3, stratify=y10, random_state=SEED)
Xtr10, Xte10 = prep(dn10.loc[tr10], dn10.loc[te10])
las10 = las_fit20(Xtr10, y10[tr10])
p10 = las10.predict_proba(Xte10)[:, 1]
auc10 = roc_auc_score(y10[te10], p10); ci10 = boot_ci(y10[te10], p10)
log(f'[B10] refit model test AUC excl. decline-only: {auc10:.3f} ({ci10[0]:.3f}-{ci10[1]:.3f}); test n={len(te10)}, events={y10[te10].sum()}')
pd.DataFrame({'analysis': ['n', 'events', 'adj_OR', 'adj_OR_lo', 'adj_OR_hi', 'adj_OR_p',
                           'test_AUC', 'test_AUC_lo', 'test_AUC_hi'],
              'value': [len(dn10), int(dn10.sicm.sum()), round(o3, 3), round(lo3, 3), round(hi3, 3), round(pv3, 4),
                        round(auc10, 3), round(ci10[0], 3), round(ci10[1], 3)]}
             ).to_csv(f'{D}/results/v6_excl_decline_only.csv', index=False)

with open(f'{D}/results/v6_review_recompute_log.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(OUT))
print('[done 106: A1+B4+B10]')
