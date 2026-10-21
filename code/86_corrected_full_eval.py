# 86_corrected_full_eval.py - complete evaluation of the primary model in the corrected cohort
# Calibration intercept/slope/ICI/Brier, baseline (age+SOFA) AUC, 1-SE model, DCA
# -*- coding: utf-8 -*-
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
import statsmodels.api as sm

SEED = 20260101
rng = np.random.RandomState(SEED)
D = '<PROJECT_ROOT>/exec'
co = pd.read_csv(f'{D}/data/cohort_main.csv')
st = pd.read_csv(f'{D}/data/cohort_dn_strict24h.csv',
                 usecols=['stay_id', 'ne_max_24h_strict', 'dobutamine_24h',
                          'milrinone_24h', 'vent_24h'])
dn = co[(co['has_hf'] == 0) & (co['congestive_heart_failure'] == 0)].merge(
    st, on='stay_id', how='left').reset_index(drop=True)

NUM = ['age','cci','sofa_24h','heart_rate_max','mbp_min','resp_rate_max','spo2_min',
       'temperature_max','lactate_max_24h','wbc_max','platelets_min','creatinine_max',
       'bun_max','bilirubin_total_max','sodium_min','potassium_max','glucose_max',
       'inr_max','ptt_max','bicarbonate_min','aniongap_max','hemoglobin_min',
       'ne_max_24h_strict','aki_stage_max']
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
y = dn['sicm'].values

tr, te = train_test_split(dn.index, test_size=0.3, stratify=y, random_state=SEED)
dtr, dte = dn.loc[tr].copy(), dn.loc[te].copy()
med = dtr[NUM].median()
dtr[NUM] = dtr[NUM].fillna(med); dte[NUM] = dte[NUM].fillna(med)
ytr, yte = dtr['sicm'].values, dte['sicm'].values
sc = StandardScaler().fit(dtr[NUM])
Xtr = dtr[FEATS].copy(); Xte = dte[FEATS].copy()
Xtr[NUM] = sc.transform(Xtr[NUM]); Xte[NUM] = sc.transform(Xte[NUM])

def cal_stats(yy, p):
    p = np.clip(p, 1e-6, 1 - 1e-6); lp = np.log(p / (1 - p))
    inter = np.asarray(sm.GLM(yy, np.ones((len(yy), 1)), family=sm.families.Binomial(),
                              offset=lp).fit().params)[0]
    prm = np.asarray(sm.Logit(yy, sm.add_constant(lp)).fit(disp=0).params)
    pc = calibration_curve(yy, p, n_bins=10, strategy='quantile')
    return inter, (prm[1] if len(prm) > 1 else np.nan), np.mean(np.abs(pc[1] - pc[0]))

def boot_auc(yy, pp, n=1000):
    a = []
    for _ in range(n):
        i = rng.randint(0, len(yy), len(yy))
        if len(np.unique(yy[i])) > 1:
            a.append(roc_auc_score(yy[i], pp[i]))
    return np.percentile(a, [2.5, 97.5])
print("[part1 loaded]")


# 1) 基线 age+SOFA
mb = LogisticRegression(max_iter=1000).fit(dtr[['age', 'sofa_24h']], ytr)
pb = mb.predict_proba(dte[['age', 'sofa_24h']])[:, 1]
# 2) LASSO lambda.min
las = LogisticRegressionCV(Cs=20, penalty='l1', solver='liblinear', cv=10,
                           scoring='roc_auc', max_iter=3000, random_state=SEED).fit(Xtr, ytr)
p1 = las.predict_proba(Xte)[:, 1]
# 3) 1-SE模型
means = las.scores_[1].mean(axis=0); ses = las.scores_[1].std(axis=0) / np.sqrt(10)
c1se = las.Cs_[np.where(means >= means.max() - ses[means.argmax()])[0][0]]
las1 = LogisticRegression(C=c1se, penalty='l1', solver='liblinear',
                          max_iter=3000, random_state=SEED).fit(Xtr, ytr)
p2 = las1.predict_proba(Xte)[:, 1]
sel1 = [f for f, c in zip(FEATS, las1.coef_[0]) if abs(c) > 1e-8]

rows = []
for name, p in [('Baseline(age+SOFA)', pb), ('LASSO lambda.min', p1), ('LASSO 1-SE', p2)]:
    inter, slope, ici = cal_stats(yte, p)
    ci = boot_auc(yte, p)
    rows.append(dict(模型=name, AUC=round(roc_auc_score(yte, p), 3),
                     AUC_lo=round(ci[0], 3), AUC_hi=round(ci[1], 3),
                     Brier=round(brier_score_loss(yte, p), 3),
                     Cal_intercept=round(inter, 3), Cal_slope=round(slope, 3),
                     ICI=round(ici, 3)))
tab = pd.DataFrame(rows)
print("=== 修正队列主模型完整评估（测试集 n=%d, 事件 %d） ===" % (len(te), yte.sum()))
print(tab.to_string(index=False))
tab.to_csv(f'{D}/results/v4_corrected_model_fulleval.csv', index=False)
print(f"1-SE变量数: {len(sel1)}: {sel1}")

# 4) DCA
ths = np.linspace(0.05, 0.5, 46)
prev = yte.mean()
dca = []
for t in ths:
    for name, p in [('LASSO lambda.min', p1), ('LASSO 1-SE', p2), ('Baseline', pb)]:
        pred = (p >= t).astype(int)
        tp = ((pred == 1) & (yte == 1)).sum(); fp = ((pred == 1) & (yte == 0)).sum()
        nb = tp / len(yte) - fp / len(yte) * t / (1 - t)
        dca.append(dict(threshold=t, model=name, net_benefit=nb))
dca = pd.DataFrame(dca)
dca_all = pd.DataFrame({'threshold': ths, 'model': 'Treat all',
                        'net_benefit': prev - (1 - prev) * ths / (1 - ths)})
dca = pd.concat([dca, dca_all, pd.DataFrame({'threshold': ths, 'model': 'Treat none',
                                             'net_benefit': 0.0})])
dca.to_csv(f'{D}/results/v4_corrected_dca.csv', index=False)
fig, ax = plt.subplots(figsize=(7, 5))
for name, c in [('LASSO lambda.min', '#d62728'), ('LASSO 1-SE', '#ff7f0e'),
                ('Baseline', '#7f7f7f'), ('Treat all', '#1f77b4'), ('Treat none', '#000000')]:
    dd = dca[dca.model == name]
    ax.plot(dd.threshold, dd.net_benefit, label=name, color=c,
            ls='--' if name in ('Treat all', 'Treat none') else '-')
ax.set_xlabel('Threshold probability'); ax.set_ylabel('Net benefit')
ax.set_ylim(-0.05, 0.3); ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f'{D}/figures/v4_corrected_dca.png', dpi=200)
print("[done]")
