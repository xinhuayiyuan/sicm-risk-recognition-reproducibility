# 96_supplement_results.py - 补报: PR-AUC + GBM比较器(Platt) + 修正队列竞争风险数据
# -*- coding: utf-8 -*-
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, psycopg2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve
import statsmodels.api as sm
import xgboost as xgb
import lightgbm as lgb

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

def boot_ci(yy, pp, n=1000):
    a = []
    for _ in range(n):
        i = rng.randint(0, len(yy), len(yy))
        if len(np.unique(yy[i])) > 1:
            a.append(roc_auc_score(yy[i], pp[i]))
    return np.percentile(a, [2.5, 97.5])

las = LogisticRegressionCV(Cs=20, penalty='l1', solver='liblinear', cv=10,
                           scoring='roc_auc', max_iter=3000, random_state=SEED).fit(Xtr, ytr)
mb = LogisticRegression(max_iter=1000).fit(dtr[['age', 'sofa_24h']], ytr)
rows = []
for name, p in [('Baseline (age+SOFA)', mb.predict_proba(dte[['age', 'sofa_24h']])[:, 1]),
                ('LASSO-logistic (lambda.min)', las.predict_proba(Xte)[:, 1])]:
    inter, slope, ici = cal_stats(yte, p)
    ci = boot_ci(yte, p)
    rows.append(dict(模型=name, AUC=round(roc_auc_score(yte, p), 3),
                     PR_AUC=round(average_precision_score(yte, p), 3),
                     AUC_lo=round(ci[0], 3), AUC_hi=round(ci[1], 3),
                     Brier=round(brier_score_loss(yte, p), 3),
                     Cal_intercept=round(inter, 3), Cal_slope=round(slope, 3), ICI=round(ici, 3)))

itr, ival = train_test_split(np.arange(len(Xtr)), test_size=0.2, stratify=ytr, random_state=SEED)
mx = xgb.train(dict(objective='binary:logistic', eval_metric='auc', max_depth=3, eta=0.03,
                    subsample=0.8, colsample_bytree=0.8, min_child_weight=20, seed=SEED),
               xgb.DMatrix(Xtr.iloc[itr], label=ytr[itr]), num_boost_round=2000,
               evals=[(xgb.DMatrix(Xtr.iloc[ival], label=ytr[ival]), 'val')],
               early_stopping_rounds=50, verbose_eval=False)
p_val = mx.predict(xgb.DMatrix(Xtr.iloc[ival]), iteration_range=(0, mx.best_iteration + 1))
p_raw = mx.predict(xgb.DMatrix(Xte), iteration_range=(0, mx.best_iteration + 1))
platt = LogisticRegression(max_iter=1000).fit(p_val.reshape(-1, 1), ytr[ival])
p_xgb = platt.predict_proba(p_raw.reshape(-1, 1))[:, 1]
ml = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.03, max_depth=3, subsample=0.8,
                        colsample_bytree=0.8, min_child_samples=20, random_state=SEED,
                        verbose=-1, n_jobs=-1)
ml.fit(Xtr.iloc[itr], ytr[itr], eval_set=[(Xtr.iloc[ival], ytr[ival])],
       callbacks=[lgb.early_stopping(50, verbose=False)])
p_val_l = ml.predict_proba(Xtr.iloc[ival])[:, 1]
p_raw_l = ml.predict_proba(Xte)[:, 1]
platt_l = LogisticRegression(max_iter=1000).fit(p_val_l.reshape(-1, 1), ytr[ival])
p_lgb = platt_l.predict_proba(p_raw_l.reshape(-1, 1))[:, 1]

for name, p in [('XGBoost (Platt-calibrated)', p_xgb), ('LightGBM (Platt-calibrated)', p_lgb)]:
    inter, slope, ici = cal_stats(yte, p)
    ci = boot_ci(yte, p)
    rows.append(dict(模型=name, AUC=round(roc_auc_score(yte, p), 3),
                     PR_AUC=round(average_precision_score(yte, p), 3),
                     AUC_lo=round(ci[0], 3), AUC_hi=round(ci[1], 3),
                     Brier=round(brier_score_loss(yte, p), 3),
                     Cal_intercept=round(inter, 3), Cal_slope=round(slope, 3), ICI=round(ici, 3)))
out = pd.DataFrame(rows)
out.to_csv(f'{D}/results/v4_corrected_model_fulleval_v2.csv', index=False)
print(out.to_string(index=False))

# ---- 竞争风险数据（修正队列） ----
conn = psycopg2.connect(dbname='mimic', host='127.0.0.1', port=5431,
                        user='postgres', password='<SET-IN-ENVIRONMENT>')
ef = pd.read_sql("""SELECT stay_id, MIN(hours_from_icu_intime) AS t_ef50
                    FROM mimiciv_echo.echo_lvef_numeric
                    WHERE lvef_value < 50 AND hours_from_icu_intime >= 0 GROUP BY stay_id""", conn)
dea = pd.read_sql("""SELECT i.stay_id, EXTRACT(EPOCH FROM (p.dod - i.intime))/3600.0 AS t_death
                     FROM mimiciv_icu.icustays i JOIN mimiciv_hosp.patients p ON i.subject_id=p.subject_id
                     WHERE p.dod IS NOT NULL AND p.dod <= i.intime + INTERVAL '28 days'""", conn)
conn.close()
dn = dn.merge(ef, on='stay_id', how='left').merge(dea, on='stay_id', how='left')
dn['t_sicm'] = np.where(dn['sicm'] == 1, dn['t_ef50'], np.nan)
def resolve(r):
    ts, td = r['t_sicm'], r['t_death']
    if pd.notna(ts) and (pd.isna(td) or ts <= td):
        return min(ts, 672.0), 1
    if pd.notna(td):
        return min(td, 672.0), 2
    return 672.0, 0
dn[['time', 'status']] = dn.apply(lambda r: pd.Series(resolve(r)), axis=1)
print(dn['status'].value_counts().to_string())
dn.to_csv(f'{D}/data/comprisk_corrected.csv', index=False)
print('[done]')
