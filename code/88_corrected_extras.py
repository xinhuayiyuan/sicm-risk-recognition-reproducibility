# 88_corrected_extras.py - additional analyses in the corrected cohort
# (1) time to echo detection / late-onset rate; (2) mortality by NE dose tertile;
# (3) SHAP; (4) DCA net-benefit range; (5) Table S3 coding sensitivity
# -*- coding: utf-8 -*-
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, psycopg2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import shap

SEED = 20260101
D = '<PROJECT_ROOT>/exec'
co = pd.read_csv(f'{D}/data/cohort_main.csv')
dn = co[(co['has_hf'] == 0) & (co['congestive_heart_failure'] == 0)].copy().reset_index(drop=True)
print(f"修正队列 n={len(dn)}, 事件={dn.sicm.sum()}")

conn = psycopg2.connect(dbname='mimic', host='127.0.0.1', port=5431,
                        user='postgres', password='<SET-IN-ENVIRONMENT>')
ef = pd.read_sql("""
    SELECT stay_id, MIN(hours_from_icu_intime) AS t_ef50
    FROM mimiciv_echo.echo_lvef_numeric
    WHERE lvef_value < 50 AND hours_from_icu_intime >= 0 GROUP BY stay_id""", conn)
dea = pd.read_sql("""
    SELECT i.stay_id, EXTRACT(EPOCH FROM (p.dod - i.intime))/3600.0 AS t_death
    FROM mimiciv_icu.icustays i JOIN mimiciv_hosp.patients p ON i.subject_id=p.subject_id
    WHERE p.dod IS NOT NULL AND p.dod <= i.intime + INTERVAL '28 days'""", conn)
conn.close()
dn = dn.merge(ef, on='stay_id', how='left').merge(dea, on='stay_id', how='left')

# ① 检出时间与迟发率（修正队列）
dn['t_sicm'] = np.where(dn['sicm'] == 1, dn['t_ef50'], np.nan)
med_t = dn.loc[dn['sicm'] == 1, 't_sicm'].median()
alive72 = dn['t_death'].isna() | (dn['t_death'] > 72)
no72 = dn['t_ef50'].isna() | (dn['t_ef50'] > 72)
lm = dn[alive72 & no72]
late = ((lm['sicm'] == 1) & (lm['t_ef50'] > 72)).sum()
print(f"① 修正队列: SICM中位检出={med_t:.1f}h | landmark n={len(lm)}, 72h后新发={late} ({late/len(lm)*100:.1f}%)")

# ② NE剂量三分位死亡率（修正队列）
ned = dn['post_max_ned'].fillna(0).values
yy = dn['mort_28d'].values.astype(float)
def tert(dsub, lab):
    n_, y_ = dsub['post_max_ned'].fillna(0).values, dsub['mort_28d'].values.astype(float)
    out = {'no NE': y_[n_ == 0].mean()}
    if (n_ > 0).sum() > 30:
        t = pd.qcut(pd.Series(n_[n_ > 0]), 3, duplicates='drop')
        for cat in t.cat.categories:
            out[str(cat)] = y_[n_ > 0][(t == cat).values].mean()
    print(f"② {lab}: " + " | ".join(f"{k}:{v:.3f}" for k, v in out.items()))
tert(dn[dn['sicm'] == 1], '新发SICM组')
tert(dn[dn['sicm'] == 0], '非SICM组')

# ③ SHAP（修正队列, 严格24h特征）
st = pd.read_csv(f'{D}/data/cohort_dn_strict24h.csv',
                 usecols=['stay_id', 'ne_max_24h_strict', 'dobutamine_24h',
                          'milrinone_24h', 'vent_24h'])
dn = dn.merge(st, on='stay_id', how='left')
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
X = dn[FEATS].fillna(dn[FEATS].median())
y = dn['sicm'].values
sc_ = StandardScaler().fit(X[NUM])
Xs = X.copy(); Xs[NUM] = sc_.transform(Xs[NUM])
mx = xgb.train(dict(objective='binary:logistic', eval_metric='auc', max_depth=3, eta=0.05,
                    subsample=0.8, colsample_bytree=0.8, min_child_weight=20, seed=SEED),
               xgb.DMatrix(Xs, label=y), num_boost_round=400, verbose_eval=False)
sv = shap.TreeExplainer(mx).shap_values(Xs)
imp = pd.DataFrame({'feature': FEATS, 'mean_abs_shap': np.abs(sv).mean(axis=0)}
                   ).sort_values('mean_abs_shap', ascending=False)
imp.to_csv(f'{D}/results/v4_corrected_shap_importance.csv', index=False)
print("③ 修正队列SHAP Top10:"); print(imp.head(10).to_string(index=False))

# ④ DCA净获益区间核验
dca = pd.read_csv(f'{D}/results/v4_corrected_dca.csv')
las = dca[dca.model == 'LASSO lambda.min'].set_index('threshold')
all_ = dca[dca.model == 'Treat all'].set_index('threshold')['net_benefit']
nb = las['net_benefit']; best = np.maximum(all_, 0.0)
pos = nb.index[nb.values > best]
print(f"④ DCA净获益为正阈值区间: {pos.min():.3f} - {pos.max():.3f}")

# ⑤ 表S3 编码敏感性
s3 = pd.DataFrame({
    '分析': ['建模队列n', 'SICM事件n(%)', '28天死亡_SICM', '28天死亡_参照',
            '调整OR(95%CI)', '调整OR_P', 'IPTW_OR(95%CI)', 'IPTW_P'],
    '仅ICD-10 I50排除(旧)': ['1,504', '676 (44.9)', '34.6%', '—',
                          '1.162 (0.919-1.470)', '0.21', '1.010 (0.786-1.297)', '0.94'],
    '全年代编码排除(主分析)': ['789', '215 (27.2)', '45.6%', '34.3%',
                            '1.659 (1.172-2.348)', '0.004', '1.502 (1.061-2.126)', '0.022']})
s3.to_csv(f'{D}/results/table_S3_coding_sensitivity.csv', index=False)
print("⑤ 表S3已生成")
print("[done]")
