# 65_score_shap.py - V3.0: Model-DN 整数化床旁评分 + SHAP解释 + 系数图（方案§6.5）
# -*- coding: utf-8 -*-
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
import xgboost as xgb
import shap

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
FEATS = [f for f in (NUM + BIN) if f != 'aids']   # aids罕见致完全分离，剔除
dn = df[df['has_hf'] == 0].reset_index(drop=True)
tr, te = train_test_split(dn.index, test_size=0.3, stratify=dn['sicm'], random_state=SEED)
dtr, dte = dn.loc[tr].copy(), dn.loc[te].copy()
med = dtr[NUM].median()
for d_ in (dtr, dte):
    d_[NUM] = d_[NUM].fillna(med)
ytr, yte = dtr['sicm'].values, dte['sicm'].values

# 1) LASSO选择（标化空间）
sc = StandardScaler().fit(dtr[NUM])
Xtr_s = dtr[FEATS].copy(); Xte_s = dte[FEATS].copy()
Xtr_s[NUM] = sc.transform(Xtr_s[NUM]); Xte_s[NUM] = sc.transform(Xte_s[NUM])
las = LogisticRegressionCV(Cs=20, penalty='l1', solver='liblinear', cv=10,
                           scoring='roc_auc', max_iter=3000, random_state=SEED)
las.fit(Xtr_s, ytr)
sel = [f for f, c in zip(FEATS, las.coef_[0]) if abs(c) > 1e-8]

# 1-SE规则：选交叉验证AUC均值不低于(max-1SE)的最强正则（最简模型）
means = las.scores_[1].mean(axis=0); ses = las.scores_[1].std(axis=0) / np.sqrt(10)
thr = means.max() - ses[means.argmax()]
cands = np.where(means >= thr)[0]
c_1se = las.Cs_[cands[0]]          # Cs降序排列，首个即最强正则
las1 = LogisticRegression(C=c_1se, penalty='l1', solver='liblinear',
                          max_iter=3000, random_state=SEED)
las1.fit(Xtr_s, ytr)
sel1 = [f for f, c in zip(FEATS, las1.coef_[0]) if abs(c) > 1e-8]
print(f"lambda.min选中{len(sel)}变量 | 1-SE规则选中{len(sel1)}变量: {sel1}")
print(f"1-SE模型测试集AUC = {roc_auc_score(yte, las1.predict_proba(Xte_s)[:, 1]):.3f}")
sel = sel1   # 评分卡基于1-SE简约变量集

# 2) 整数评分: z标化重拟合（连续变量按SD、二分类按有无，量纲可比）-> 中位单位取整
cont_in_sel = [v for v in sel if v in NUM]
mu, sd_ = dtr[cont_in_sel].mean(), dtr[cont_in_sel].std().replace(0, 1)
Ztr = dtr[sel].astype(float).copy(); Zte = dte[sel].astype(float).copy()
Ztr[cont_in_sel] = (Ztr[cont_in_sel] - mu) / sd_
Zte[cont_in_sel] = (Zte[cont_in_sel] - mu) / sd_
m = sm.Logit(ytr, sm.add_constant(Ztr)).fit(disp=0)
beta = m.params.drop('const')
unit = np.median(np.abs(beta))       # 中位|系数|=1分（稳健单位，分值可读）
pts = (beta / unit).round().astype(int)
pts[pts == 0] = np.sign(beta[pts == 0]).astype(int)   # 避免0分退化
score_te = (Zte * pts).sum(axis=1)
auc_score = roc_auc_score(yte, score_te)
auc_lasso = roc_auc_score(yte, las.predict_proba(Xte_s)[:, 1])
print(f"整数评分({len(sel)}项, z标化): 测试集AUC={auc_score:.3f} | LASSO AUC={auc_lasso:.3f} | 衰减={auc_lasso-auc_score:+.3f}")
ptab = pd.DataFrame({'变量': sel, '系数_z': beta.round(3).values, '分值': pts.values})
ptab.to_csv(f'{D}/results/v3_DN_integer_score.csv', index=False)
print(ptab.to_string(index=False))

# 3) 系数图（列线图替代）
fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(sel))))
o = ptab.reindex(ptab['系数_z'].abs().sort_values().index)
ax.barh(o['变量'], o['系数_z'], color=['#d62728' if v > 0 else '#1f77b4' for v in o['系数_z']])
ax.set_xlabel('Logistic coefficient (z-standardized refit)')
ax.set_title('Model-DN selected predictors')
ax.grid(alpha=0.3, axis='x')
fig.tight_layout(); fig.savefig(f'{D}/figures/v3_DN_coefficients.png', dpi=200)

# 4) SHAP (XGBoost)
itr, ival = train_test_split(np.arange(len(Xtr_s)), test_size=0.2, stratify=ytr, random_state=SEED)
mx = xgb.train(dict(objective='binary:logistic', eval_metric='auc', max_depth=3, eta=0.03,
                    subsample=0.8, colsample_bytree=0.8, min_child_weight=20, seed=SEED),
               xgb.DMatrix(Xtr_s.iloc[itr], label=ytr[itr]), num_boost_round=2000,
               evals=[(xgb.DMatrix(Xtr_s.iloc[ival], label=ytr[ival]), 'val')],
               early_stopping_rounds=50, verbose_eval=False)
expl = shap.TreeExplainer(mx)
sv = expl.shap_values(Xte_s)
imp = pd.DataFrame({'feature': FEATS, 'mean_abs_shap': np.abs(sv).mean(axis=0)}).sort_values(
    'mean_abs_shap', ascending=False)
imp.to_csv(f'{D}/results/v3_DN_shap_importance.csv', index=False)
print("\nSHAP Top10:"); print(imp.head(10).to_string(index=False))
shap.summary_plot(sv, Xte_s, feature_names=FEATS, show=False, max_display=15)
plt.tight_layout(); plt.savefig(f'{D}/figures/v3_DN_shap_summary.png', dpi=200, bbox_inches='tight')
print("\n[done]")
