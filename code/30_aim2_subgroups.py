# 30_aim2_subgroups.py
# 阶段B-步骤3：Aim 2 细化内部分析（方案V2.0 §六）
# 1) 新发表型内临床亚组异质性（OOF预测 -> 亚组AUC/校准斜率 + bootstrap CI + BH-FDR）
# 2) 时间稳定性（leave-one-era-out by anchor_year_group）
# 3) 敏感性S1：4套结局定义重跑LASSO
# -*- coding: utf-8 -*-
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve
import statsmodels.api as sm

SEED = 20260101
rng = np.random.RandomState(SEED)
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

def prep_fit(dtrain, dtest):
    med = dtrain[NUM].median()
    Xtr = dtrain[FEATS].copy(); Xte = dtest[FEATS].copy()
    Xtr[NUM] = Xtr[NUM].fillna(med); Xte[NUM] = Xte[NUM].fillna(med)
    sc = StandardScaler().fit(Xtr[NUM])
    Xtr[NUM] = sc.transform(Xtr[NUM]); Xte[NUM] = sc.transform(Xte[NUM])
    return Xtr, Xte

def lasso_fit(Xtr, ytr):
    m = LogisticRegressionCV(Cs=10, penalty='l1', solver='liblinear', cv=5,
                             scoring='roc_auc', max_iter=3000, random_state=SEED)
    m.fit(Xtr, ytr); return m

# ---------- 1) OOF预测（5折） ----------
oof = np.zeros(len(dn))
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
for tr, va in skf.split(dn, y):
    Xtr, Xva = prep_fit(dn.iloc[tr], dn.iloc[va])
    oof[va] = lasso_fit(Xtr, y[tr]).predict_proba(Xva)[:, 1]
print(f"OOF AUC = {roc_auc_score(y, oof):.3f}")

def boot_ci_auc(yy, pp, n=500):
    a = []
    for _ in range(n):
        i = rng.randint(0, len(yy), len(yy))
        if len(np.unique(yy[i])) < 2: continue
        a.append(roc_auc_score(yy[i], pp[i]))
    return np.percentile(a, [2.5, 97.5]) if a else (np.nan, np.nan)

def cal_slope(yy, pp):
    pp = np.clip(pp, 1e-6, 1 - 1e-6); lp = np.log(pp / (1 - pp))
    try:
        return sm.Logit(yy, sm.add_constant(lp)).fit(disp=0).params[1]
    except Exception:
        return np.nan

# ---------- 亚组异质性 ----------
subgroups = {
    '年龄': [('年龄<65', dn['age_grp'] == '<65'), ('65-74', dn['age_grp'] == '65-74'),
            ('>=75', dn['age_grp'] == '>=75')],
    'SOFA': [('SOFA<8', dn['sofa_grp'] == 0), ('SOFA>=8', dn['sofa_grp'] == 1)],
    '休克(24h NE)': [('未用NE', dn['shock_24h'] == 0), ('已用NE', dn['shock_24h'] == 1)],
    'AKI': [(f'AKI {k}', dn['aki_grp'] == k) for k in [0, 1, 2, 3]],
    '机械通气': [('未通气', dn['invasive_vent'] == 0), ('有创通气', dn['invasive_vent'] == 1)],
}
rows = []
auc_all = roc_auc_score(y, oof)
for dim, lst in subgroups.items():
    for lab, mask in lst:
        m = mask.values
        yy, pp = y[m], oof[m]
        if m.sum() < 60 or yy.sum() < 20:
            rows.append(dict(维度=dim, 亚组=lab, n=int(m.sum()), 事件=int(yy.sum()),
                             AUC=np.nan, AUC_lo=np.nan, AUC_hi=np.nan, 校准斜率=np.nan, p_vs_整体=np.nan))
            continue
        ci = boot_ci_auc(yy, pp)
        # bootstrap检验：亚组AUC与整体AUC差异
        diffs = []
        for _ in range(500):
            i = rng.randint(0, len(yy), len(yy))
            j = rng.randint(0, len(y), len(y))
            if len(np.unique(yy[i])) < 2 or len(np.unique(y[j])) < 2: continue
            diffs.append(roc_auc_score(yy[i], pp[i]) - roc_auc_score(y[j], oof[j]))
        diffs = np.array(diffs)
        p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
        rows.append(dict(维度=dim, 亚组=lab, n=int(m.sum()), 事件=int(yy.sum()),
                         AUC=round(roc_auc_score(yy, pp), 3), AUC_lo=round(ci[0], 3),
                         AUC_hi=round(ci[1], 3), 校准斜率=round(cal_slope(yy, pp), 3),
                         p_vs_整体=round(p, 4)))
sub = pd.DataFrame(rows)
# BH-FDR
pv = sub['p_vs_整体'].dropna().sort_values()
q = pv * len(pv) / (np.arange(len(pv)) + 1)
q = q[::-1].cummin()[::-1]
sub.loc[pv.index, 'q_FDR'] = q.values.round(4)
print(f"\n整体OOF AUC = {auc_all:.3f}\n=== 亚组异质性 ===")
print(sub.to_string(index=False))
sub.to_csv(f'{D}/results/aim2_subgroup_heterogeneity.csv', index=False)
print("[done part1]")


# ---------- 2) 时间稳定性：leave-one-era-out ----------
eras = dn['anchor_year_group'].unique()
era_rows = []
for e in sorted(eras):
    mtr = dn['anchor_year_group'] != e
    mva = ~mtr
    if mva.sum() < 50 or y[mva.values].sum() < 10:
        era_rows.append(dict(验证年代=e, n=int(mva.sum()), 事件=int(y[mva.values].sum()),
                             AUC=np.nan, 训练AUC=np.nan))
        continue
    Xtr, Xva = prep_fit(dn[mtr], dn[mva])
    m = lasso_fit(Xtr, y[mtr.values])
    pva = m.predict_proba(Xva)[:, 1]
    ptr = m.predict_proba(Xtr)[:, 1]
    era_rows.append(dict(验证年代=e, n=int(mva.sum()), 事件=int(y[mva.values].sum()),
                         AUC=round(roc_auc_score(y[mva.values], pva), 3),
                         训练AUC=round(roc_auc_score(y[mtr.values], ptr), 3)))
era_tab = pd.DataFrame(era_rows)
print("\n=== 时间稳定性（leave-one-era-out） ===")
print(era_tab.to_string(index=False))
era_tab.to_csv(f'{D}/results/aim2_temporal_era_cv.csv', index=False)

# ---------- 3) 敏感性S1：4套结局定义 ----------
sens_rows = []
for outc, lab in [('sicm_ef40', 'EF<40%'), ('sicm_drop15', 'EF<50%或降≥15pp'),
                  ('sicm_img', '纯影像'), ('sicm_defB', '复合定义B(EF+血管活性药)')]:
    yy = dn[outc].values
    tr, te = train_test_split(np.arange(len(dn)), test_size=0.3, stratify=yy, random_state=SEED)
    Xtr, Xte = prep_fit(dn.iloc[tr], dn.iloc[te])
    m = lasso_fit(Xtr, yy[tr])
    pte = m.predict_proba(Xte)[:, 1]
    sens_rows.append(dict(结局定义=lab, 事件率=round(yy.mean(), 3),
                          AUC_test=round(roc_auc_score(yy[te], pte), 3),
                          AUC_lo=round(boot_ci_auc(yy[te], pte)[0], 3),
                          AUC_hi=round(boot_ci_auc(yy[te], pte)[1], 3)))
sens = pd.DataFrame(sens_rows)
print("\n=== 敏感性S1：结局定义稳健性 ===")
print(sens.to_string(index=False))
sens.to_csv(f'{D}/results/aim2_sensitivity_S1_definitions.csv', index=False)
print("\n[done]")
