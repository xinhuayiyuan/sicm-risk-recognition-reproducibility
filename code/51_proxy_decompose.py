# 51_proxy_decompose.py
# 迁移失败归因分解：同一Model-DN-e在MIMIC内部预测 ①真实SICM(影像) ②MIMIC版代理终点
# 若②同样低 -> 构念不匹配主导；若②高 -> 数据库迁移问题主导
# -*- coding: utf-8 -*-
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

SEED = 20260101
D = '<PROJECT_ROOT>/exec'
df = pd.read_csv(f'{D}/data/cohort_main.csv')
dn = df[df['has_hf'] == 0].copy()

COMORB = ['myocardial_infarct','peripheral_vascular_disease','cerebrovascular_disease',
          'dementia','chronic_pulmonary_disease','rheumatic_disease','mild_liver_disease',
          'severe_liver_disease','renal_disease','malignant_cancer','metastatic_solid_tumor','aids']
for c in COMORB + ['invasive_vent']:
    dn[c] = dn[c].fillna(0).astype(int)
MFEATS = ['age','sofa_24h','heart_rate_max','mbp_min','resp_rate_max','spo2_min',
          'lactate_max_24h','wbc_max','platelets_min','hemoglobin_min','bilirubin_total_max',
          'creatinine_max','sodium_min','potassium_max','glucose_max','inr_max','ptt_max',
          'bicarbonate_min','aniongap_max'] + COMORB + ['invasive_vent']

# MIMIC版代理终点（与eICU同构）: 无HF + cTn↑ + NE使用
dn['proxy_sicm_mimic'] = ((dn['trop_gt014'] == 1) & (dn['post_max_ned'].fillna(0) > 0)).astype(int)

X = dn[MFEATS].fillna(dn[MFEATS].median())
sc = StandardScaler().fit(X)
model = LogisticRegressionCV(Cs=10, penalty='l1', solver='liblinear', cv=5,
                             scoring='roc_auc', max_iter=3000, random_state=SEED)
model.fit(sc.transform(X), dn['sicm'].values)
p = model.predict_proba(sc.transform(X))[:, 1]

print("=== Model-DN-e（训练目标=影像SICM）在MIMIC内部的表观表现 ===")
print(f"对 真实影像SICM      : AUC = {roc_auc_score(dn['sicm'], p):.3f} (事件率 {dn['sicm'].mean():.3f})")
print(f"对 MIMIC版代理终点   : AUC = {roc_auc_score(dn['proxy_sicm_mimic'], p):.3f} (事件率 {dn['proxy_sicm_mimic'].mean():.3f})")
print("\n解读: 若两者接近eICU迁移AUC(0.551) -> 构念不匹配主导；")
print("      若MIMIC代理AUC明显高于0.551 -> 数据库迁移(测量时窗/人群)亦有贡献")
print("[done]")
