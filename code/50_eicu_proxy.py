# 50_eicu_proxy.py
# 阶段B-步骤5：eICU辅助验证（方案V2.0 §八）
# V1: Model-DN-e（两库共有、非循环特征子集）冻结迁移 -> eICU代理终点
#     代理终点 = 无既存心衰 + cTnI峰值>0.04 + NE使用（血流动力学表型proxy）
#     报告 AUC/校准截距/斜率/Brier + 截距再校准
# V3: 路径一致性 = 心肌损伤proxy(无HF+cTnI↑) -> NE使用(中介) -> 28天死亡, g-computation
# -*- coding: utf-8 -*-
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import psycopg2
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss
import statsmodels.api as sm

SEED = 20260101
rng = np.random.RandomState(SEED)
D = '<PROJECT_ROOT>/exec'

# ---------- 0. 读取eICU ----------
conn = psycopg2.connect(dbname='eicu', host='127.0.0.1', port=5431,
                        user='postgres', password='<SET-IN-ENVIRONMENT>')
ei = pd.read_sql("SELECT * FROM eicu_crd.temp_final_export", conn)
conn.close()
ei['age_num'] = pd.to_numeric(ei['age'].astype(str).str.replace('> 89', '90'), errors='coerce')
ei = ei[(ei['age_num'] >= 18) & (ei['age_num'] <= 90) & (ei['los_icu_days'] >= 1)].copy()

trop_cols = [c for c in ei.columns if c.startswith('troponin_i_') and c != 'troponin_i_first']
ei['trop_max_72h'] = ei[trop_cols + ['troponin_i_first']].max(axis=1)
ei = ei[ei['trop_max_72h'].notna()].copy()          # 需有cTn测定（对应MIMIC可评估）
ei['ne_used'] = (ei['norepi_max_rate'].fillna(0) > 0).astype(int)
# 代理终点（V1）
ei['proxy_sicm'] = ((ei['congestive_heart_failure'] == 0) &
                    (ei['trop_max_72h'] > 0.04) & (ei['ne_used'] == 1)).astype(int)
print(f"eICU代理队列 n={len(ei)}, 代理SICM={ei.proxy_sicm.sum()} ({ei.proxy_sicm.mean()*100:.1f}%), "
      f"28天死亡={ei.mortality_28.mean():.3f}")

# ---------- 1. 共有非循环特征映射 ----------
FEAT_MAP = {   # MIMIC列 -> eICU列（均为入ICU早期值；时间窗差异在局限性声明）
    'age': 'age_num', 'sofa_24h': 'sofa_total', 'heart_rate_max': 'heart_rate_first',
    'mbp_min': 'mbp_first', 'resp_rate_max': 'resp_rate_first', 'spo2_min': 'spo2_first',
    'lactate_max_24h': 'lactate_first', 'wbc_max': 'wbc_first', 'platelets_min': 'platelet_first',
    'hemoglobin_min': 'hemoglobin_first', 'bilirubin_total_max': 'bilirubin_total_first',
    'creatinine_max': 'creatinine_first', 'sodium_min': 'sodium_first',
    'potassium_max': 'potassium_first', 'glucose_max': 'glucose_first',
    'inr_max': 'inr_first', 'ptt_max': 'ptt_first', 'bicarbonate_min': 'bicarbonate_first',
    'aniongap_max': 'aniongap_first',
    'myocardial_infarct': 'myocardial_infarct',
    'peripheral_vascular_disease': 'peripheral_vascular_disease',
    'cerebrovascular_disease': 'cerebrovascular_disease', 'dementia': 'dementia',
    'chronic_pulmonary_disease': 'chronic_pulmonary_disease',
    'rheumatic_disease': 'rheumatic_disease', 'mild_liver_disease': 'mild_liver_disease',
    'severe_liver_disease': 'severe_liver_disease', 'renal_disease': 'renal_disease',
    'malignant_cancer': 'malignant_cancer', 'metastatic_solid_tumor': 'metastatic_solid_tumor',
    'aids': 'aids', 'invasive_vent': 'invasivevent'}
MFEATS = list(FEAT_MAP.keys())

# ---------- 2. 在MIMIC新发队列训练Model-DN-e ----------
df = pd.read_csv(f'{D}/data/cohort_main.csv')
dn = df[df['has_hf'] == 0].copy()
for c in ['invasive_vent']:
    dn[c] = dn[c].fillna(0).astype(int)
COMORB = ['myocardial_infarct','peripheral_vascular_disease','cerebrovascular_disease',
          'dementia','chronic_pulmonary_disease','rheumatic_disease','mild_liver_disease',
          'severe_liver_disease','renal_disease','malignant_cancer','metastatic_solid_tumor','aids']
for c in COMORB:
    dn[c] = dn[c].fillna(0).astype(int)

med_mimic = dn[MFEATS].median()
Xm = dn[MFEATS].fillna(med_mimic)
ym = dn['sicm'].values
sc = StandardScaler().fit(Xm)
model = LogisticRegressionCV(Cs=10, penalty='l1', solver='liblinear', cv=5,
                             scoring='roc_auc', max_iter=3000, random_state=SEED)
model.fit(sc.transform(Xm), ym)
print(f"Model-DN-e (MIMIC内部, 表观AUC): {roc_auc_score(ym, model.predict_proba(sc.transform(Xm))[:,1]):.3f}")

# ---------- 3. 冻结迁移至eICU ----------
Xe = ei[[FEAT_MAP[f] for f in MFEATS]].copy()
Xe.columns = MFEATS
for c in COMORB + ['invasive_vent']:
    Xe[c] = Xe[c].fillna(0)
Xe = Xe.fillna(med_mimic)                # 用MIMIC中位数插补（部署惯例）
pe = model.predict_proba(sc.transform(Xe))[:, 1]
ye = ei['proxy_sicm'].values

def boot_auc_ci(yy, pp, n=500):
    a = []
    for _ in range(n):
        i = rng.randint(0, len(yy), len(yy))
        if len(np.unique(yy[i])) < 2: continue
        a.append(roc_auc_score(yy[i], pp[i]))
    return np.percentile(a, [2.5, 97.5])

auc_e = roc_auc_score(ye, pe); ci_e = boot_auc_ci(ye, pe)
lp = np.log(np.clip(pe, 1e-6, 1-1e-6) / (1 - np.clip(pe, 1e-6, 1-1e-6)))
inter = sm.GLM(ye, np.ones((len(ye), 1)), family=sm.families.Binomial(), offset=lp).fit().params[0]
_prm = sm.Logit(ye, sm.add_constant(lp)).fit(disp=0).params
slope = _prm[1] if len(_prm) > 1 else np.nan
brier = brier_score_loss(ye, pe)
# 截距再校准（V4）
pe_rec = 1 / (1 + np.exp(-(lp + inter)))
print(f"\n=== V1 模型迁移（冻结） ===")
print(f"AUC={auc_e:.3f} (95%CI {ci_e[0]:.3f}-{ci_e[1]:.3f}), Brier={brier:.3f}")
print(f"校准截距={inter:.3f}, 校准斜率={slope:.3f}")
print(f"V4 截距再校准后 Brier={brier_score_loss(ye, pe_rec):.3f}")
pd.DataFrame({'metric': ['AUC', 'AUC_lo', 'AUC_hi', 'Brier', 'cal_intercept', 'cal_slope',
                         'Brier_recalibrated'],
              'value': [auc_e, ci_e[0], ci_e[1], brier, inter, slope,
                        brier_score_loss(ye, pe_rec)]}
             ).to_csv(f'{D}/results/aim_eicu_V1_transport.csv', index=False)
print("[done part1]")

# ---------- 4. V3 路径一致性：心肌损伤proxy -> NE使用 -> 28天死亡 ----------
# 暴露: 无既存心衰 + cTnI峰值>0.04（避免与中介NE构成循环，暴露不含NE）
ei['mi_proxy'] = ((ei['congestive_heart_failure'] == 0) &
                  (ei['trop_max_72h'] > 0.04)).astype(int)
CONF_E = ['age_num', 'sofa_total', 'lactate_first', 'invasivevent', 'aki_score']
sub = ei.dropna(subset=['mortality_28']).copy()
for c in ['lactate_first']:
    sub[c] = sub[c].fillna(sub[c].median())
sub['invasivevent'] = sub['invasivevent'].fillna(0)
sub['aki_score'] = sub['aki_score'].fillna(0)
# 与MIMIC口径对齐：中介=NE使用(二分类)；混杂=年龄/SOFA/乳酸/通气/AKI
dE = sub[['mi_proxy', 'ne_used', 'mortality_28'] + CONF_E].copy()
dE = dE.rename(columns={'mi_proxy': 'sicm', 'ne_used': 'M', 'mortality_28': 'Y'})
print(f"\nV3分析集 n={len(dE)}, 暴露率={dE.sicm.mean():.3f}, NE使用={dE.M.mean():.3f}, "
      f"死亡={dE.Y.mean():.3f}")

def mediation_gcomp(d, n_boot=300):
    def fit_effects(dd):
        n = len(dd)
        MC = sm.add_constant(pd.concat([dd['sicm'], dd[CONF_E]], axis=1).astype(float))
        mfit = sm.Logit(dd['M'], MC).fit(disp=0)
        YC = sm.add_constant(pd.DataFrame({'sicm': dd['sicm'], 'M': dd['M']}
                                          ).join(dd[CONF_E]).astype(float))
        yfit = sm.Logit(dd['Y'], YC).fit(disp=0)
        def sim_m(xv):
            Xd = pd.DataFrame({'const': 1.0, 'sicm': float(xv)}, index=range(n)).join(dd[CONF_E].astype(float))
            return (rng.random(n) < mfit.predict(Xd)).astype(float)
        def pred_y(xv, Mv):
            Yd = pd.DataFrame({'const': 1.0, 'sicm': float(xv), 'M': Mv}).join(dd[CONF_E].astype(float))
            return yfit.predict(Yd)
        M0 = sim_m(0.0); M1 = sim_m(1.0)
        y00 = pred_y(0.0, M0).mean(); y11 = pred_y(1.0, M1).mean(); y10 = pred_y(1.0, M0).mean()
        te = y11 - y00; nde = y10 - y00; nie = y11 - y10
        return te, nde, nie, (nie / te if abs(te) > 1e-9 else np.nan)
    pt = fit_effects(d)
    boots = []
    for _ in range(n_boot):
        try:
            boots.append(fit_effects(d.iloc[rng.randint(0, len(d), len(d))].reset_index(drop=True)))
        except Exception:
            continue
    ci = np.nanpercentile(np.array(boots), [2.5, 97.5], axis=0)
    return pt, ci

pt_e, ci_e3 = mediation_gcomp(dE, n_boot=300)
print("=== V3 eICU路径一致性（暴露=心肌损伤proxy, 中介=NE使用, 结局=28天死亡） ===")
print(f"TE  = {pt_e[0]:+.4f} (95%CI {ci_e3[0,0]:+.4f} ~ {ci_e3[1,0]:+.4f})")
print(f"NDE = {pt_e[1]:+.4f} (95%CI {ci_e3[0,1]:+.4f} ~ {ci_e3[1,1]:+.4f})")
print(f"NIE = {pt_e[2]:+.4f} (95%CI {ci_e3[0,2]:+.4f} ~ {ci_e3[1,2]:+.4f})")
print(f"PM  = {pt_e[3]*100:.1f}% (95%CI {ci_e3[0,3]*100:.1f}% ~ {ci_e3[1,3]*100:.1f}%)")
pd.DataFrame({'effect': ['TE', 'NDE', 'NIE', 'PM'], 'estimate': pt_e,
              'ci_lo': ci_e3[0], 'ci_hi': ci_e3[1]}
             ).to_csv(f'{D}/results/aim_eicu_V3_mediation.csv', index=False)
print("\n[done]")

