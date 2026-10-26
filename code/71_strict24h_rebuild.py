# 71_strict24h_rebuild.py - 泄漏修正：严格0-24h特征重建Model-DN
# 替换: dobutamine/milrinone/vent -> 24h内启用标志; post_max_ned -> 0-24h NE最大速率(可溯源)
# 其余特征均为mimiciv_derived.first_day_*（窗口可信）
# -*- coding: utf-8 -*-
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, psycopg2
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
import statsmodels.api as sm

SEED = 20260101
rng = np.random.RandomState(SEED)
D = '<PROJECT_ROOT>/exec'
co = pd.read_csv(f'{D}/data/cohort_main.csv')
dn = co[co['has_hf'] == 0].copy().reset_index(drop=True)
ids = [int(s) for s in dn['stay_id']]

conn = psycopg2.connect(dbname='mimic', host='127.0.0.1', port=5431,
                        user='postgres', password='<SET-IN-ENVIRONMENT>')
cur = conn.cursor()
cur.execute("""
    SELECT n.stay_id, MAX(n.vaso_rate) AS ne_max_24h_strict
    FROM mimiciv_derived.norepinephrine n JOIN mimiciv_icu.icustays i ON n.stay_id=i.stay_id
    WHERE n.stay_id = ANY(%s) AND n.starttime <= i.intime + INTERVAL '24 hours'
    GROUP BY n.stay_id""", (ids,))
ne24 = pd.DataFrame(cur.fetchall(), columns=['stay_id', 'ne_max_24h_strict'])
flags = {}
for tab in ['dobutamine', 'milrinone']:
    cur.execute(f"""
        SELECT DISTINCT t.stay_id FROM mimiciv_derived.{tab} t
        JOIN mimiciv_icu.icustays i ON t.stay_id=i.stay_id
        WHERE t.stay_id = ANY(%s) AND t.starttime <= i.intime + INTERVAL '24 hours'""", (ids,))
    flags[tab] = {r[0] for r in cur.fetchall()}
cur.execute("""
    SELECT DISTINCT v.stay_id FROM mimiciv_derived.ventilation v
    JOIN mimiciv_icu.icustays i ON v.stay_id=i.stay_id
    WHERE v.stay_id = ANY(%s) AND v.ventilation_status ILIKE '%%Invasive%%'
      AND v.starttime <= i.intime + INTERVAL '24 hours'""", (ids,))
flags['vent'] = {r[0] for r in cur.fetchall()}
cur.close(); conn.close()

dn = dn.merge(ne24, on='stay_id', how='left')
dn['ne_max_24h_strict'] = dn['ne_max_24h_strict'].fillna(0)
dn['dobutamine_24h'] = dn['stay_id'].isin(flags['dobutamine']).astype(int)
dn['milrinone_24h'] = dn['stay_id'].isin(flags['milrinone']).astype(int)
dn['vent_24h'] = dn['stay_id'].isin(flags['vent']).astype(int)
print(f"严格24h版本使用率: dobutamine {dn.dobutamine_24h.mean():.3f} (原{dn.dobutamine.mean():.3f}), "
      f"milrinone {dn.milrinone_24h.mean():.3f} (原{dn.milrinone.mean():.3f}), "
      f"vent {dn.vent_24h.mean():.3f} (原{dn.invasive_vent.mean():.3f})")
dn.to_csv(f'{D}/data/cohort_dn_strict24h.csv', index=False)
print("[part1 done]")

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
FEATS = NUM + BIN
y = dn['sicm'].values

def prep(a, b):
    med = a[NUM].median()
    Xa = a[FEATS].copy(); Xb = b[FEATS].copy()
    Xa[NUM] = Xa[NUM].fillna(med); Xb[NUM] = Xb[NUM].fillna(med)
    sc = StandardScaler().fit(Xa[NUM])
    Xa[NUM] = sc.transform(Xa[NUM]); Xb[NUM] = sc.transform(Xb[NUM])
    return Xa, Xb

def las_fit(X, yy):
    m = LogisticRegressionCV(Cs=20, penalty='l1', solver='liblinear', cv=10,
                             scoring='roc_auc', max_iter=3000, random_state=SEED)
    m.fit(X, yy); return m

def boot_ci(yy, pp, n=1000):
    a = []
    for _ in range(n):
        i = rng.randint(0, len(yy), len(yy))
        if len(np.unique(yy[i])) < 2: continue
        a.append(roc_auc_score(yy[i], pp[i]))
    return np.percentile(a, [2.5, 97.5])

tr, te = train_test_split(dn.index, test_size=0.3, stratify=y, random_state=SEED)
Xtr, Xte = prep(dn.loc[tr], dn.loc[te])
m = las_fit(Xtr, y[tr])
p = m.predict_proba(Xte)[:, 1]
ci = boot_ci(y[te], p)
pp = np.clip(p, 1e-6, 1-1e-6); lp = np.log(pp/(1-pp))
prm = np.asarray(sm.Logit(y[te], sm.add_constant(lp)).fit(disp=0).params)
pc = calibration_curve(y[te], p, n_bins=10, strategy='quantile')
sel = [f for f, c in zip(FEATS, m.coef_[0]) if abs(c) > 1e-8]
print(f"\n=== 严格24h Model-DN（测试集, 同一切分） ===")
print(f"AUC={roc_auc_score(y[te], p):.3f} (95%CI {ci[0]:.3f}-{ci[1]:.3f}), "
      f"Brier={brier_score_loss(y[te], p):.3f}, 校准斜率={prm[1]:.3f}, "
      f"ICI={np.mean(np.abs(pc[1]-pc[0])):.3f}")
print(f"选中{len(sel)}变量, dobutamine_24h在列: {'dobutamine_24h' in sel}, "
      f"milrinone_24h在列: {'milrinone_24h' in sel}")

oof = np.zeros(len(dn))
for tr2, va in StratifiedKFold(5, shuffle=True, random_state=SEED).split(dn, y):
    Xa, Xb = prep(dn.iloc[tr2], dn.iloc[va])
    mm = las_fit(Xa, y[tr2])
    oof[va] = mm.predict_proba(Xb)[:, 1]
print(f"严格24h OOF AUC = {roc_auc_score(y, oof):.3f}  (原含泄漏版本OOF: 0.743)")
pd.DataFrame({'版本': ['含泄漏(原,测试集)', '严格24h(测试集)', '严格24h(OOF)'],
              'AUC': [0.768, round(roc_auc_score(y[te], p), 3), round(roc_auc_score(y, oof), 3)],
              'AUC_lo': [0.727, round(ci[0], 3), np.nan],
              'AUC_hi': [0.811, round(ci[1], 3), np.nan]}
             ).to_csv(f'{D}/results/strict24h_model_comparison.csv', index=False)
print("[done]")

