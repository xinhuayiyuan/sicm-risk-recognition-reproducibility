# 61_s7_iptw.py
# S7 超声选择偏倚校正：以全部Sepsis-3队列建"接受超声评估"PS模型 -> 稳定化IPTW
# 应用于: ①新发SICM->28天死亡OR ②定义B->28天死亡OR ③Model-DN OOF加权AUC
# -*- coding: utf-8 -*-
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import psycopg2
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm

SEED = 20260101
rng = np.random.RandomState(SEED)
D = '<PROJECT_ROOT>/exec'

# ---- 全Sepsis队列（含不可评估者） ----
conn = psycopg2.connect(dbname='mimic', host='127.0.0.1', port=5431,
                        user='postgres', password='<SET-IN-ENVIRONMENT>')
full = pd.read_sql("""SELECT stay_id, age, gender, sofa_24h, heart_rate_max, mbp_min,
                      resp_rate_max, spo2_min, lactate_max_24h, wbc_max, creatinine_max,
                      invasive_vent, cci, evaluable, has_hf, has_acs, mort_28d
                      FROM mimiciv_echo.sicm_analysis_dataset
                      WHERE age>=18 AND age<=90 AND has_acs=0""", conn)
conn.close()
print(f"全Sepsis队列(18-90,排ACS) n={len(full)}, 可评估={full.evaluable.sum()} ({full.evaluable.mean()*100:.1f}%)")

PSF = ['age','sofa_24h','heart_rate_max','mbp_min','resp_rate_max','spo2_min',
       'lactate_max_24h','wbc_max','creatinine_max','cci']
Xps = full[PSF].copy()
for c in PSF:
    Xps[c + '_miss'] = Xps[c].isna().astype(int)
    Xps[c] = Xps[c].fillna(Xps[c].median())
Xps['invasive_vent'] = full['invasive_vent'].fillna(0)
Xps['gender_f'] = (full['gender'] == 'F').astype(int)
sc = StandardScaler().fit(Xps)
psm = LogisticRegression(max_iter=2000).fit(sc.transform(Xps), full['evaluable'])
ps = psm.predict_proba(sc.transform(Xps))[:, 1]
prev = full['evaluable'].mean()
w = np.where(full['evaluable'] == 1, prev / ps, (1 - prev) / (1 - ps))
lo, hi = np.percentile(w[full['evaluable'] == 1], [1, 99])
full['sw'] = np.where(full['evaluable'] == 1, np.clip(w, lo, hi), 1.0)
print(f"稳定化IPTW（可评估者截尾1-99%: {lo:.2f}-{hi:.2f}）, 均值={full.loc[full.evaluable==1,'sw'].mean():.2f}")

# PS模型区分度
print(f"PS模型AUC(可评估~协变量) = {roc_auc_score(full['evaluable'], ps):.3f}")

# ---- 加权关联分析（可评估、无HF史队列） ----
co = pd.read_csv(f'{D}/data/cohort_main.csv')
co = co.merge(full[['stay_id', 'sw']], on='stay_id', how='left')
dn = co[co['has_hf'] == 0].copy()
dn['lact_f'] = dn['lactate_max_24h'].fillna(dn['lactate_max_24h'].median())
CONF = ['age', 'sofa_24h', 'cci', 'lact_f', 'invasive_vent', 'aki_stage_max']
for c in ['invasive_vent', 'aki_stage_max']:
    dn[c] = dn[c].fillna(0)

def w_logit(dsub, xcol, label):
    X = sm.add_constant(pd.concat([dsub[xcol], dsub[CONF]], axis=1).astype(float))
    mu = sm.Logit(dsub['mort_28d'], X).fit(disp=0)
    mw = sm.GLM(dsub['mort_28d'], X, family=sm.families.Binomial(),
                freq_weights=dsub['sw']).fit()
    ci_u = np.exp(mu.conf_int().loc[xcol]); ci_w = np.exp(mw.conf_int().loc[xcol])
    print(f"{label}: 未加权OR={np.exp(mu.params[xcol]):.3f} ({ci_u[0]:.3f}-{ci_u[1]:.3f}) | "
          f"IPTW加权OR={np.exp(mw.params[xcol]):.3f} ({ci_w[0]:.3f}-{ci_w[1]:.3f}), P={mw.pvalues[xcol]:.4f}")
    return dict(exposure=label,
                OR_unw=round(np.exp(mu.params[xcol]), 3), OR_iptw=round(np.exp(mw.params[xcol]), 3),
                lo_iptw=round(ci_w[0], 3), hi_iptw=round(ci_w[1], 3), p_iptw=float(f"{mw.pvalues[xcol]:.2e}"))


# ---- 加权OOF AUC（Model-DN，5折OOF + sample_weight评估） ----
NUM = ['age','cci','sofa_24h','heart_rate_max','mbp_min','resp_rate_max','spo2_min',
       'temperature_max','lactate_max_24h','wbc_max','platelets_min','creatinine_max',
       'bun_max','bilirubin_total_max','sodium_min','potassium_max','glucose_max',
       'inr_max','ptt_max','bicarbonate_min','aniongap_max','hemoglobin_min',
       'post_max_ned','aki_stage_max']
COMORB = ['myocardial_infarct','peripheral_vascular_disease','cerebrovascular_disease',
          'dementia','chronic_pulmonary_disease','rheumatic_disease','mild_liver_disease',
          'diabetes_without_cc','diabetes_with_cc','renal_disease','malignant_cancer',
          'severe_liver_disease','metastatic_solid_tumor','aids']
BIN = ['gender_f','race_black','race_other'] + COMORB + \
      ['invasive_vent','dobutamine','milrinone','rrt_firstday','lactate_miss','bili_miss']
dn['gender_f'] = (dn['gender'] == 'F').astype(int)
rc2 = dn['race'].astype(str).str.upper()
dn['race_black'] = rc2.str.contains('BLACK').astype(int)
dn['race_other'] = (~rc2.str.contains('WHITE|BLACK')).astype(int)
for c in COMORB + ['invasive_vent','dobutamine','milrinone','rrt_firstday']:
    dn[c] = dn[c].fillna(0).astype(int)
FEATS = NUM + BIN
dn = dn.reset_index(drop=True)
yv = dn['sicm'].values
oof = np.zeros(len(dn))
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
for tr, va in skf.split(dn, yv):
    a, b = dn.iloc[tr], dn.iloc[va]
    med = a[NUM].median()
    Xa = a[FEATS].copy(); Xb = b[FEATS].copy()
    Xa[NUM] = Xa[NUM].fillna(med); Xb[NUM] = Xb[NUM].fillna(med)
    scx = StandardScaler().fit(Xa[NUM])
    Xa[NUM] = scx.transform(Xa[NUM]); Xb[NUM] = scx.transform(Xb[NUM])
    m = LogisticRegressionCV(Cs=10, penalty='l1', solver='liblinear', cv=5,
                             scoring='roc_auc', max_iter=3000, random_state=SEED)
    m.fit(Xa, yv[tr])
    oof[va] = m.predict_proba(Xb)[:, 1]
auc_uw = roc_auc_score(yv, oof)
auc_w = roc_auc_score(yv, oof, sample_weight=dn['sw'].values)
print(f"\nModel-DN OOF AUC: 未加权={auc_uw:.3f} | IPTW加权={auc_w:.3f}")
pd.DataFrame({'metric': ['OOF_AUC_unweighted', 'OOF_AUC_IPTW', 'PS_model_AUC'],
              'value': [round(auc_uw,3), round(auc_w,3), round(roc_auc_score(full['evaluable'], ps),3)]}
             ).to_csv(f'{D}/results/v3_S7_weighted_auc.csv', index=False)
print("\n[done]")

r1 = w_logit(dn, 'sicm', '新发SICM(表型一)')
r2 = w_logit(dn, 'sicm_defB', '新发SICM失代偿(定义B)')
pd.DataFrame([r1, r2]).to_csv(f'{D}/results/v3_S7_iptw_association.csv', index=False)
print("[done part1]")
