# 63_recalibration.py - V3.0: AKI1期 / SOFA>=8 亚组特异性再校准（方案§6.1流程）
# 交叉拟合(2折)避免再校准过拟合; 报告再校准前后 ICI / Brier / 校准截距 / 斜率
# -*- coding: utf-8 -*-
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
import statsmodels.api as sm

SEED = 20260101
D = '<PROJECT_ROOT>/exec'
df = pd.read_csv(f'{D}/data/cohort_main.csv')
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
df['gender_f'] = (df['gender'] == 'F').astype(int)
rc = df['race'].astype(str).str.upper()
df['race_black'] = rc.str.contains('BLACK').astype(int)
df['race_other'] = (~rc.str.contains('WHITE|BLACK')).astype(int)
for c in COMORB + ['invasive_vent','dobutamine','milrinone','rrt_firstday']:
    df[c] = df[c].fillna(0).astype(int)
FEATS = NUM + BIN
dn = df[df['has_hf'] == 0].reset_index(drop=True)
y = dn['sicm'].values

oof = np.zeros(len(dn))
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
for tr, va in skf.split(dn, y):
    a, b = dn.iloc[tr], dn.iloc[va]
    med = a[NUM].median()
    Xa = a[FEATS].copy(); Xb = b[FEATS].copy()
    Xa[NUM] = Xa[NUM].fillna(med); Xb[NUM] = Xb[NUM].fillna(med)
    sc = StandardScaler().fit(Xa[NUM])
    Xa[NUM] = sc.transform(Xa[NUM]); Xb[NUM] = sc.transform(Xb[NUM])
    m = LogisticRegressionCV(Cs=10, penalty='l1', solver='liblinear', cv=5,
                             scoring='roc_auc', max_iter=3000, random_state=SEED)
    m.fit(Xa, y[tr]); oof[va] = m.predict_proba(Xb)[:, 1]

def ici(yy, pp):
    pc = calibration_curve(yy, pp, n_bins=8, strategy='quantile')
    return float(np.mean(np.abs(pc[1] - pc[0])))

def cal_metrics(yy, pp):
    pp = np.clip(pp, 1e-6, 1-1e-6); lp = np.log(pp/(1-pp))
    inter = np.asarray(sm.GLM(yy, np.ones((len(yy),1)), family=sm.families.Binomial(), offset=lp).fit().params)[0]
    prm = np.asarray(sm.Logit(yy, sm.add_constant(lp)).fit(disp=0).params)
    slope = prm[1] if len(prm) > 1 else np.nan
    return inter, slope

# 亚组内2折交叉拟合的再校准
def subgroup_recal(mask, name):
    idx = np.where(mask.values)[0]
    yy, pp = y[idx], oof[idx]
    # 再校准前
    i0, s0 = cal_metrics(yy, pp)
    before = dict(ICI=round(ici(yy, pp), 3), Brier=round(brier_score_loss(yy, pp), 3),
                  Cal_intercept=round(i0, 3), Cal_slope=round(s0, 3))
    # 交叉拟合再校准: 一半拟合 y~lp, 另一半应用
    pp_rec = np.zeros_like(pp)
    kf = StratifiedKFold(2, shuffle=True, random_state=SEED)
    lp_full = np.log(np.clip(pp, 1e-6, 1-1e-6) / (1 - np.clip(pp, 1e-6, 1-1e-6)))
    for a, b in kf.split(pp.reshape(-1, 1), yy):
        rec = LogisticRegression(max_iter=1000).fit(lp_full[a].reshape(-1, 1), yy[a])
        pp_rec[b] = rec.predict_proba(lp_full[b].reshape(-1, 1))[:, 1]
    i1, s1 = cal_metrics(yy, pp_rec)
    after = dict(ICI=round(ici(yy, pp_rec), 3), Brier=round(brier_score_loss(yy, pp_rec), 3),
                 Cal_intercept=round(i1, 3), Cal_slope=round(s1, 3))
    print(f"\n=== {name} (n={len(idx)}, 事件{yy.sum()}) ===")
    print(f"  再校准前: {before}")
    print(f"  再校准后: {after}")
    return dict(亚组=name, n=len(idx), AUC=round(roc_auc_score(yy, pp), 3),
                ICI_前=before['ICI'], ICI_后=after['ICI'],
                Brier_前=before['Brier'], Brier_后=after['Brier'],
                斜率_前=before['Cal_slope'], 斜率_后=after['Cal_slope'])

rows = [subgroup_recal(dn['aki_grp'] == 1, 'AKI 1期'),
        subgroup_recal(dn['sofa_grp'] == 1, 'SOFA>=8'),
        subgroup_recal(dn['aki_grp'] == 0, 'AKI 0期(对照)')]
pd.DataFrame(rows).to_csv(f'{D}/results/v3_subgroup_recalibration.csv', index=False)
print("\n[done]")
