# 97_rigorous_fixes.py - 核查数据级修复
# ①landmark-24h分析 ②敏感性表型死亡OR ③LASSO标准化系数表 ④评分卡均值/SD/截距
# ⑤Table S3英文版(补齐参照死亡率) ⑥Fig4 SHAP可读标签
# -*- coding: utf-8 -*-
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
import xgboost as xgb
import shap

SEED = 20260101
rng = np.random.RandomState(SEED)
D = '<PROJECT_ROOT>/exec'
co = pd.read_csv(f'{D}/data/cohort_main.csv')
st = pd.read_csv(f'{D}/data/cohort_dn_strict24h.csv',
                 usecols=['stay_id', 'ne_max_24h_strict', 'dobutamine_24h',
                          'milrinone_24h', 'vent_24h'])
dn0 = co[(co['has_hf'] == 0) & (co['congestive_heart_failure'] == 0)].merge(
    st, on='stay_id', how='left').reset_index(drop=True)
cr = pd.read_csv(f'{D}/data/comprisk_corrected.csv')[['stay_id', 't_ef50']]
dn0 = dn0.merge(cr, on='stay_id', how='left')

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
dn0['gender_f'] = (dn0['gender'] == 'F').astype(int)
rc = dn0['race'].astype(str).str.upper()
dn0['race_black'] = rc.str.contains('BLACK').astype(int)
dn0['race_other'] = (~rc.str.contains('WHITE|BLACK')).astype(int)
for c in COMORB + ['rrt_firstday']:
    dn0[c] = dn0[c].fillna(0).astype(int)
FEATS = [f for f in (NUM + BIN) if f != 'aids']

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
        if len(np.unique(yy[i])) > 1:
            a.append(roc_auc_score(yy[i], pp[i]))
    return np.percentile(a, [2.5, 97.5])

# ---- 主模型（修正队列，供运输与系数表） ----
y0 = dn0['sicm'].values
tr, te = train_test_split(dn0.index, test_size=0.3, stratify=y0, random_state=SEED)
Xtr0, Xte0 = prep(dn0.loc[tr], dn0.loc[te])
m0 = las_fit(Xtr0, y0[tr])

# ============ ① landmark-24h ============
lm = dn0[(dn0['t_ef50'].isna()) | (dn0['t_ef50'] > 24)].copy()
ylm = ((lm['sicm'] == 1) & (lm['t_ef50'] > 24)).astype(int).values
print(f"① landmark-24h队列 n={len(lm)}, 24h后新发SICM={ylm.sum()} ({ylm.mean()*100:.1f}%)")
tr2, te2 = train_test_split(np.arange(len(lm)), test_size=0.3, stratify=ylm, random_state=SEED)
Xa, Xb = prep(lm.iloc[tr2], lm.iloc[te2])
ylm_tr, ylm_te = ylm[tr2], ylm[te2]
# (a) 全队列冻结模型运输
p_trans = m0.predict_proba(prep(lm.iloc[tr2], lm.iloc[te2])[1])[:, 1]
auc_trans = roc_auc_score(ylm_te, p_trans)
ci_trans = boot_ci(ylm_te, p_trans)
# (b) landmark内重训
m_lm = las_fit(Xa, ylm_tr)
p_lm = m_lm.predict_proba(Xb)[:, 1]
auc_lm = roc_auc_score(ylm_te, p_lm)
ci_lm = boot_ci(ylm_te, p_lm)
print(f"   运输(冻结模型): AUC={auc_trans:.3f} ({ci_trans[0]:.3f}-{ci_trans[1]:.3f})")
print(f"   landmark内重训: AUC={auc_lm:.3f} ({ci_lm[0]:.3f}-{ci_lm[1]:.3f})")
pd.DataFrame({'analysis': ['landmark_cohort_n', 'events_after_24h', 'event_rate',
                           'transport_AUC', 'transport_lo', 'transport_hi',
                           'refit_AUC', 'refit_lo', 'refit_hi'],
              'value': [len(lm), int(ylm.sum()), round(ylm.mean(), 3),
                        round(auc_trans, 3), round(ci_trans[0], 3), round(ci_trans[1], 3),
                        round(auc_lm, 3), round(ci_lm[0], 3), round(ci_lm[1], 3)]}
             ).to_csv(f'{D}/results/v4_landmark24h.csv', index=False)

# ============ ② 敏感性表型死亡关联（修正队列） ============
dn0['lact_f'] = dn0['lactate_max_24h'].fillna(dn0['lactate_max_24h'].median())
CONF = ['age', 'sofa_24h', 'cci', 'lact_f', 'invasive_vent', 'aki_stage_max']
dn0['aki_stage_max'] = dn0['aki_stage_max'].fillna(0)
rows = []
for outc, lab in [('sicm_ef40', 'LVEF <40%'),
                  ('sicm_drop15', 'LVEF drop ≥15 pp',
                   ),
                  ('sicm_img', 'Imaging-only (no troponin requirement)'),
                  ('sicm_defB', 'Decompensated (EF + vasopressor escalation)')]:
    n_ev = int(dn0[outc].sum())
    if n_ev < 30:
        rows.append(dict(phenotype=lab, events=n_ev, OR=np.nan, lo=np.nan, hi=np.nan, p=np.nan))
        continue
    X = sm.add_constant(pd.concat([dn0[outc], dn0[CONF]], axis=1).astype(float))
    m = sm.Logit(dn0['mort_28d'].astype(float), X).fit(disp=0)
    ci = np.exp(m.conf_int().loc[outc])
    rows.append(dict(phenotype=lab, events=n_ev, OR=round(np.exp(m.params[outc]), 3),
                     lo=round(ci[0], 3), hi=round(ci[1], 3),
                     p=float(f"{m.pvalues[outc]:.2e}")))
sens = pd.DataFrame(rows)
sens.to_csv(f'{D}/results/v4_sensitivity_phenotypes_mortality.csv', index=False)
print("② 敏感性表型死亡关联:")
print(sens.to_string(index=False))

# ============ ③ LASSO标准化系数表（1-SE主模型） ============
coef = pd.Series(m0.coef_[0], index=FEATS)
coef = coef[coef.abs() > 1e-8].sort_values(key=abs, ascending=False)
top = pd.DataFrame({'feature': coef.index, 'std_coef': coef.values.round(3)})
top.to_csv(f'{D}/results/v4_lasso_std_coefficients.csv', index=False)
print("③ 主模型标准化系数Top10:")
print(top.head(10).to_string(index=False))
# MI单变量OR（对照佐证）
m_uni = sm.Logit(y0, sm.add_constant(dn0['myocardial_infarct'].astype(float)).values).fit(disp=0)
ci_u = np.exp(m_uni.conf_int())
print(f"   MI单变量OR={np.exp(m_uni.params[1]):.2f} ({ci_u[1,0]:.2f}-{ci_u[1,1]:.2f})")

# ============ ④ 评分卡均值/SD/截距 ============
sel = list(top['feature'])
cont = [v for v in sel if v in NUM]
med = dn0.loc[tr, NUM].median()
mu = dn0.loc[tr, cont].mean(); sd_ = dn0.loc[tr, cont].std().replace(0, 1)
dtr_z = dn0.loc[tr, sel].astype(float).copy()
dtr_z[cont] = dtr_z[cont].fillna(dn0.loc[tr, cont].median())
dtr_z[cont] = (dtr_z[cont] - mu) / sd_
mm = sm.Logit(dn0.loc[tr, 'sicm'], sm.add_constant(dtr_z)).fit(disp=0)
beta = mm.params.drop('const')
unit = np.median(np.abs(beta))
pts = (beta / unit).round().astype(int)
pts[pts == 0] = np.sign(beta[pts == 0]).astype(int)
sc_rows = []
for f in sel:
    sc_rows.append(dict(feature=f, mean=round(mu[f], 3) if f in cont else np.nan,
                        sd=round(sd_[f], 3) if f in cont else np.nan,
                        coef_z=round(beta[f], 3), points=int(pts[f])))
sc_df = pd.DataFrame(sc_rows)
sc_df.loc[len(sc_df)] = dict(feature='(Intercept)', mean=np.nan, sd=np.nan,
                             coef_z=round(mm.params['const'], 3), points=np.nan)
sc_df.to_csv(f'{D}/results/v4_score_card_full.csv', index=False)
print("④ 评分卡(含均值/SD/截距)已生成, 截距=", round(mm.params['const'], 3))

# ============ ⑤ Table S3英文版（补齐两口径参照死亡率） ============
leg = co[co['has_hf'] == 0]   # I50-only队列（旧口径）
s3 = pd.DataFrame({
    'Analysis': ['Modelling cohort, n', 'SICM events, n (%)',
                 '28-day mortality, SICM, %', '28-day mortality, reference, %',
                 'Adjusted OR (95% CI)', 'Adjusted OR, P',
                 'IPW OR (95% CI)', 'IPW OR, P'],
    'ICD-10 I50 only (legacy)': [
        f"{len(leg):,}", f"{int(leg.sicm.sum())} ({leg.sicm.mean()*100:.1f})",
        f"{leg.loc[leg.sicm==1,'mort_28d'].mean()*100:.1f}",
        f"{leg.loc[leg.sicm==0,'mort_28d'].mean()*100:.1f}",
        '1.162 (0.919–1.470)', '0.21', '1.010 (0.786–1.297)', '0.94'],
    'Era-spanning (primary)': [
        f"{len(dn0):,}", f"{int(dn0.sicm.sum())} ({dn0.sicm.mean()*100:.1f})",
        f"{dn0.loc[dn0.sicm==1,'mort_28d'].mean()*100:.1f}",
        f"{dn0.loc[dn0.sicm==0,'mort_28d'].mean()*100:.1f}",
        '1.659 (1.172–2.348)', '0.004', '1.502 (1.061–2.126)', '0.022']})
s3.to_csv(f'{D}/results/table_S3_coding_sensitivity_en.csv', index=False)
print("⑤ Table S3英文版:")
print(s3.to_string(index=False))

# ============ ⑥ Fig4 SHAP可读标签 ============
FEAT_EN = {'myocardial_infarct': 'Prior myocardial infarction',
           'ptt_max': 'PTT max', 'glucose_max': 'Glucose max',
           'heart_rate_max': 'Heart rate max', 'temperature_max': 'Temperature max',
           'hemoglobin_min': 'Haemoglobin min', 'platelets_min': 'Platelets min',
           'sodium_min': 'Sodium min', 'aniongap_max': 'Anion gap max',
           'resp_rate_max': 'Respiratory rate max', 'age': 'Age',
           'wbc_max': 'WBC max', 'potassium_max': 'Potassium max',
           'inr_max': 'INR max', 'bilirubin_total_max': 'Bilirubin max',
           'creatinine_max': 'Creatinine max', 'bun_max': 'BUN max',
           'lactate_max_24h': 'Lactate max', 'spo2_min': 'SpO2 min',
           'mbp_min': 'MAP min', 'sofa_24h': 'SOFA score', 'cci': 'Charlson index',
           'ne_max_24h_strict': 'Peak norepinephrine (24 h)',
           'aki_stage_max': 'AKI stage', 'bicarbonate_min': 'Bicarbonate min',
           'dobutamine_24h': 'Dobutamine (24 h)', 'milrinone_24h': 'Milrinone (24 h)',
           'vent_24h': 'Invasive ventilation (24 h)', 'rrt_firstday': 'RRT (day 1)',
           'gender_f': 'Female sex', 'race_black': 'Black race', 'race_other': 'Other race',
           'lactate_miss': 'Lactate missing', 'bili_miss': 'Bilirubin missing',
           'peripheral_vascular_disease': 'Peripheral vascular disease',
           'cerebrovascular_disease': 'Cerebrovascular disease', 'dementia': 'Dementia',
           'chronic_pulmonary_disease': 'Chronic pulmonary disease',
           'rheumatic_disease': 'Rheumatic disease', 'mild_liver_disease': 'Mild liver disease',
           'diabetes_without_cc': 'Diabetes', 'diabetes_with_cc': 'Diabetes with complications',
           'renal_disease': 'Renal disease', 'malignant_cancer': 'Malignancy',
           'severe_liver_disease': 'Severe liver disease',
           'metastatic_solid_tumor': 'Metastatic tumour', 'aids': 'AIDS'}
X0 = dn0[FEATS].fillna(dn0[FEATS].median())
sc0 = StandardScaler().fit(X0[NUM])
Xs0 = X0.copy(); Xs0[NUM] = sc0.transform(Xs0[NUM])
mx = xgb.train(dict(objective='binary:logistic', eval_metric='auc', max_depth=3, eta=0.05,
                    subsample=0.8, colsample_bytree=0.8, min_child_weight=20, seed=SEED),
               xgb.DMatrix(Xs0, label=y0), num_boost_round=400, verbose_eval=False)
sv = shap.TreeExplainer(mx).shap_values(Xs0)
labels = [FEAT_EN.get(f, f) for f in FEATS]
plt.figure(figsize=(8, 6))
shap.summary_plot(sv, Xs0, feature_names=labels, show=False, max_display=15)
plt.tight_layout()
plt.savefig(f'{D}/figures/fig4_shap_corrected.png', dpi=200, bbox_inches='tight')
plt.close('all')
print("⑥ Fig4 SHAP可读标签已重绘")
print('[done]')
