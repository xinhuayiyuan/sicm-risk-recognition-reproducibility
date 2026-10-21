# 74_iptw_robust.py
# 核查决议补充②：IPTW稳健三明治SE（HC1）+ 全流程bootstrap（PS重建含于每次抽样）
# -*- coding: utf-8 -*-
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import psycopg2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

SEED = 20260101
rng = np.random.RandomState(SEED)
D = '<PROJECT_ROOT>/exec'

conn = psycopg2.connect(dbname='mimic', host='127.0.0.1', port=5431,
                        user='postgres', password='<SET-IN-ENVIRONMENT>')
full = pd.read_sql("""SELECT stay_id, age, gender, sofa_24h, heart_rate_max, mbp_min,
                      resp_rate_max, spo2_min, lactate_max_24h, wbc_max, creatinine_max,
                      invasive_vent, cci, evaluable, has_hf, has_acs
                      FROM mimiciv_echo.sicm_analysis_dataset
                      WHERE age>=18 AND age<=90 AND has_acs=0""", conn)
conn.close()

PSF = ['age','sofa_24h','heart_rate_max','mbp_min','resp_rate_max','spo2_min',
       'lactate_max_24h','wbc_max','creatinine_max','cci']

def ps_weights(df_full):
    Xps = df_full[PSF].copy()
    for c in PSF:
        Xps[c] = Xps[c].fillna(Xps[c].median())
    Xps['invasive_vent'] = df_full['invasive_vent'].fillna(0)
    Xps['gender_f'] = (df_full['gender'] == 'F').astype(int)
    sc = StandardScaler().fit(Xps)
    m = LogisticRegression(max_iter=1500).fit(sc.transform(Xps), df_full['evaluable'])
    ps = np.clip(m.predict_proba(sc.transform(Xps))[:, 1], 0.02, 0.98)
    prev = df_full['evaluable'].mean()
    w = np.where(df_full['evaluable'] == 1, prev / ps, 1.0)
    we = w[df_full['evaluable'] == 1]
    lo, hi = np.percentile(we, [1, 99])
    w[df_full['evaluable'] == 1] = np.clip(we, lo, hi)
    return np.asarray(w)   # ndarray，避免重采样后stay_id重复导致的索引冲突

co = pd.read_csv(f'{D}/data/cohort_main.csv')
dn = co[co['has_hf'] == 0].copy()
dn['lact_f'] = dn['lactate_max_24h'].fillna(dn['lactate_max_24h'].median())
CONF = ['age', 'sofa_24h', 'cci', 'lact_f', 'invasive_vent', 'aki_stage_max']
for c in ['invasive_vent', 'aki_stage_max']:
    dn[c] = dn[c].fillna(0)

# ---- 点估计 + HC1稳健SE ----
full['sw'] = ps_weights(full)
dn = dn.merge(full[['stay_id', 'sw']], on='stay_id', how='left')

def fit_iptw(dsub, xcol):
    d2 = dsub.dropna(subset=['sw', 'mort_28d', xcol] + CONF).copy()
    d2 = d2[d2['sw'] > 0]
    X = sm.add_constant(pd.concat([d2[xcol], d2[CONF]], axis=1).astype(float))
    m = sm.GLM(d2['mort_28d'].astype(float), X, family=sm.families.Binomial(),
               freq_weights=d2['sw']).fit(cov_type='HC1')
    return np.exp(m.params[xcol]), np.exp(m.conf_int().loc[xcol]), m.pvalues[xcol]

rows = []
for xc, lab in [('sicm', '新发SICM(表型一)'), ('sicm_defB', '新发SICM失代偿(定义B)')]:
    orr, ci, p = fit_iptw(dn, xc)
    print(f"{lab}: IPTW OR={orr:.3f} (稳健95%CI {ci[0]:.3f}-{ci[1]:.3f}), P={p:.4f}")
    rows.append(dict(exposure=lab, OR=round(orr, 3), lo_robust=round(ci[0], 3),
                     hi_robust=round(ci[1], 3), p_robust=float(f"{p:.2e}")))

# ---- 全流程bootstrap（300次，每次重建PS与权重） ----
def boot_or(xcol, n_boot=300):
    ors = []
    ev_ids = dn['stay_id'].values
    for _ in range(n_boot):
        try:
            fb = full.iloc[rng.randint(0, len(full), len(full))].reset_index(drop=True)
            fb['sw'] = ps_weights(fb)
            db = dn.drop(columns=['sw'], errors='ignore').iloc[
                rng.randint(0, len(dn), len(dn))].copy()
            db = db.merge(fb[['stay_id', 'sw']].groupby('stay_id').first().reset_index(),
                          on='stay_id', how='left')
            db['sw'] = db['sw'].fillna(db['sw'].median())
            db = db.dropna(subset=['sw', 'mort_28d', xcol] + CONF)
            db = db[db['sw'] > 0]
            X = sm.add_constant(pd.concat([db[xcol], db[CONF]], axis=1).astype(float))
            m = sm.GLM(db['mort_28d'].astype(float), X, family=sm.families.Binomial(),
                       freq_weights=db['sw']).fit()
            ors.append(np.exp(m.params[xcol]))
        except Exception:
            continue
    return np.percentile(ors, [2.5, 97.5])

for xc, lab in [('sicm', '新发SICM(表型一)'), ('sicm_defB', '新发SICM失代偿(定义B)')]:
    lo, hi = boot_or(xc)
    print(f"{lab}: 全流程bootstrap 95%CI {lo:.3f}-{hi:.3f}")
    for r in rows:
        if r['exposure'] == lab:
            r['lo_bootfull'] = round(lo, 3); r['hi_bootfull'] = round(hi, 3)

pd.DataFrame(rows).to_csv(f'{D}/results/v3_S7_iptw_robust.csv', index=False)
print("\n对照(61脚本freq_weights近似CI): 表型一 0.795-1.316 | 定义B 0.883-1.752")
print("[done]")
