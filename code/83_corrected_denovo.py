# 83_corrected_denovo.py - 表型修正: "新发"= 无ICD-10 I50心衰(hf_history) 且 无Charlson CHF(ICD-9 428.x+I50)
# 量化修正对主分析的影响: 队列规模/事件数/主模型AUC/死亡关联
# -*- coding: utf-8 -*-
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm

SEED = 20260101
rng = np.random.RandomState(SEED)
D = '<PROJECT_ROOT>/exec'
co = pd.read_csv(f'{D}/data/cohort_main.csv')

print("=== 表型归属变化（原has_hf二分 -> 修正三分） ===")
for _, r in co.iterrows.__self__.groupby(['has_hf', 'congestive_heart_failure']).size().reset_index(name='n').iterrows():
    pass
tab = co.groupby(['has_hf', 'congestive_heart_failure']).agg(
    n=('sicm', 'size'), sicm=('sicm', 'sum')).reset_index()
print(tab.to_string(index=False))

# 修正后新发队列: has_hf=0 且 Charlson CHF=0
dn2 = co[(co['has_hf'] == 0) & (co['congestive_heart_failure'] == 0)].copy().reset_index(drop=True)
print(f"\n修正后新发建模队列: n={len(dn2)} (原1504), 事件={dn2.sicm.sum()} (原676), "
      f"事件率={dn2.sicm.mean():.3f}, 28天死亡={dn2.mort_28d.mean():.3f}")
# 被重新归类者: has_hf=0 但 CHF=1
moved = co[(co['has_hf'] == 0) & (co['congestive_heart_failure'] == 1)]
print(f"重新归类为'非新发'(ICD-9期CHF漏标): n={len(moved)}, 其中SICM={moved.sicm.sum()}")

# ---- 修正队列主模型（严格24h特征, 同流程） ----
NUM = ['age','cci','sofa_24h','heart_rate_max','mbp_min','resp_rate_max','spo2_min',
       'temperature_max','lactate_max_24h','wbc_max','platelets_min','creatinine_max',
       'bun_max','bilirubin_total_max','sodium_min','potassium_max','glucose_max',
       'inr_max','ptt_max','bicarbonate_min','aniongap_max','hemoglobin_min',
       'aki_stage_max']
COMORB = ['myocardial_infarct','peripheral_vascular_disease','cerebrovascular_disease',
          'dementia','chronic_pulmonary_disease','rheumatic_disease','mild_liver_disease',
          'diabetes_without_cc','diabetes_with_cc','renal_disease','malignant_cancer',
          'severe_liver_disease','metastatic_solid_tumor','aids']
BIN = ['gender_f','race_black','race_other'] + COMORB + \
      ['invasive_vent','rrt_firstday','lactate_miss','bili_miss']
dn2['gender_f'] = (dn2['gender'] == 'F').astype(int)
rc = dn2['race'].astype(str).str.upper()
dn2['race_black'] = rc.str.contains('BLACK').astype(int)
dn2['race_other'] = (~rc.str.contains('WHITE|BLACK')).astype(int)
for c in COMORB + ['rrt_firstday', 'invasive_vent']:
    dn2[c] = dn2[c].fillna(0).astype(int)
FEATS = [f for f in (NUM + BIN) if f != 'aids']
y = dn2['sicm'].values

def prep(a, b):
    med = a[NUM].median()
    Xa = a[FEATS].copy(); Xb = b[FEATS].copy()
    Xa[NUM] = Xa[NUM].fillna(med); Xb[NUM] = Xb[NUM].fillna(med)
    sc = StandardScaler().fit(Xa[NUM])
    Xa[NUM] = sc.transform(Xa[NUM]); Xb[NUM] = sc.transform(Xb[NUM])
    return Xa, Xb

def las_fit(X, yy):
    m = LogisticRegressionCV(Cs=10, penalty='l1', solver='liblinear', cv=5,
                             scoring='roc_auc', max_iter=3000, random_state=SEED)
    m.fit(X, yy); return m

tr, te = train_test_split(dn2.index, test_size=0.3, stratify=y, random_state=SEED)
Xtr, Xte = prep(dn2.loc[tr], dn2.loc[te])
m = las_fit(Xtr, y[tr])
p = m.predict_proba(Xte)[:, 1]
aucs = []
for _ in range(1000):
    i = rng.randint(0, len(te), len(te))
    if len(np.unique(y[te][i])) > 1:
        aucs.append(roc_auc_score(y[te][i], p[i]))
ci = np.percentile(aucs, [2.5, 97.5])
oof = np.zeros(len(dn2))
for tr2, va in StratifiedKFold(5, shuffle=True, random_state=SEED).split(dn2, y):
    Xa, Xb = prep(dn2.iloc[tr2], dn2.iloc[va])
    oof[va] = las_fit(Xa, y[tr2]).predict_proba(Xb)[:, 1]
print(f"\n=== 修正队列主模型 ===")
print(f"测试集 AUC={roc_auc_score(y[te], p):.3f} (95%CI {ci[0]:.3f}-{ci[1]:.3f}) | "
      f"OOF AUC={roc_auc_score(y, oof):.3f}")

# ---- 修正队列死亡关联 ----
dn2['lact_f'] = dn2['lactate_max_24h'].fillna(dn2['lactate_max_24h'].median())
CONF = ['age','sofa_24h','cci','lact_f','invasive_vent','aki_stage_max']
dn2['aki_stage_max'] = dn2['aki_stage_max'].fillna(0)
X2 = sm.add_constant(pd.concat([dn2['sicm'], dn2[CONF]], axis=1).astype(float))
ma = sm.Logit(dn2['mort_28d'], X2).fit(disp=0)
ci2 = np.exp(ma.conf_int().loc['sicm'])
print(f"\n=== 修正队列 新发SICM->28天死亡 调整OR ===")
print(f"OR={np.exp(ma.params['sicm']):.3f} (95%CI {ci2[0]:.3f}-{ci2[1]:.3f}), P={ma.pvalues['sicm']:.4f}")
print(f"粗死亡率: SICM {dn2.loc[dn2.sicm==1,'mort_28d'].mean():.3f} vs 非SICM {dn2.loc[dn2.sicm==0,'mort_28d'].mean():.3f}")
pd.DataFrame({'指标': ['修正队列n', '事件', '事件率', '测试集AUC', 'AUC_lo', 'AUC_hi', 'OOF_AUC',
                     '调整OR', 'OR_lo', 'OR_hi', 'OR_P'],
              '值': [len(dn2), int(dn2.sicm.sum()), round(dn2.sicm.mean(), 3),
                    round(roc_auc_score(y[te], p), 3), round(ci[0], 3), round(ci[1], 3),
                    round(roc_auc_score(y, oof), 3), round(np.exp(ma.params['sicm']), 3),
                    round(ci2[0], 3), round(ci2[1], 3), float(f"{ma.pvalues['sicm']:.2e}")]}
             ).to_csv(f'{D}/results/v4_corrected_denovo_model.csv', index=False)
print("[done]")
