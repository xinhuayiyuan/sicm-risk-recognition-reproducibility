# 40_aim3_mediation.py
# 阶段B-步骤4：Aim 3 血流动力学作用路径（方案V2.0 §七）
# X=新发SICM, Y=28天死亡, C=基线混杂(age/sofa_24h/cci/lactate/vent/aki)
# 中介: M1=vaso_new_upgrade(二分类), M2=post_max_ned(连续, log1p)
# 方法: g-computation自然效应(RD尺度) + bootstrap 1000 CI; 剂量-反应RCS
# -*- coding: utf-8 -*-
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
from patsy import dmatrix

SEED = 20260101
rng = np.random.RandomState(SEED)
D = '<PROJECT_ROOT>/exec'
df = pd.read_csv(f'{D}/data/cohort_main.csv')
dn = df[df['has_hf'] == 0].copy().reset_index(drop=True)
dn['log_ned'] = np.log1p(dn['post_max_ned'].fillna(0))
CONF = ['age', 'sofa_24h', 'cci', 'lactate_max_24h', 'invasive_vent', 'aki_stage_max']
dn['lactate_max_24h'] = dn['lactate_max_24h'].fillna(dn['lactate_max_24h'].median())
for c in ['aki_stage_max', 'invasive_vent']:
    dn[c] = dn[c].fillna(0)

X = dn['sicm'].values.astype(float)
Y = dn['mort_28d'].values.astype(float)
C = sm.add_constant(dn[CONF].astype(float))

# ---------- 0. 调整关联（主效应） ----------
m_adj = sm.Logit(Y, sm.add_constant(pd.concat([dn['sicm'], dn[CONF]], axis=1).astype(float))).fit(disp=0)
or_sicm = np.exp(m_adj.params['sicm']); ci = np.exp(m_adj.conf_int().loc['sicm'])
print(f"=== 新发SICM与28天死亡（调整OR） ===\nOR={or_sicm:.3f} (95%CI {ci[0]:.3f}-{ci[1]:.3f}), P={m_adj.pvalues['sicm']:.4f}")

# ---------- 1. g-computation 中介分析 ----------
def mediation_effects(d, X, Y, mediator, mtype, n_boot=1000):
    """返回 TE, NDE, NIE, PM (RD尺度) 及bootstrap CI"""
    def fit_effects(dd, XX, YY):
        Cc = sm.add_constant(dd[CONF].astype(float))
        MC = sm.add_constant(pd.concat([dd['sicm'], dd[CONF]], axis=1).astype(float))
        if mtype == 'bin':
            mfit = sm.Logit(dd[mediator], MC).fit(disp=0)
        else:
            mfit = sm.OLS(dd[mediator], MC).fit()
        YC = sm.add_constant(pd.DataFrame({'sicm': dd['sicm'], 'M': dd[mediator]}
                                          ).join(dd[CONF]).astype(float))
        yfit = sm.Logit(YY, YC).fit(disp=0)
        n = len(dd)
        def sim_m(xval):
            Xd = pd.DataFrame({'const': 1.0, 'sicm': xval}, index=range(n)).join(dd[CONF].astype(float))
            if mtype == 'bin':
                pr = mfit.predict(Xd)
                return (rng.random(n) < pr).astype(float)
            else:
                mu = mfit.predict(Xd)
                return np.maximum(mu + rng.normal(0, np.sqrt(mfit.mse_resid), n), 0)
        def pred_y(xval, Mv):
            Yd = pd.DataFrame({'const': 1.0, 'sicm': float(xval), 'M': Mv}).join(dd[CONF].astype(float))
            return yfit.predict(Yd)
        M0 = sim_m(0.0); M1 = sim_m(1.0)
        y00 = pred_y(0.0, M0).mean(); y11 = pred_y(1.0, M1).mean()
        y10 = pred_y(1.0, M0).mean()
        te = y11 - y00; nde = y10 - y00; nie = y11 - y10
        pm = nie / te if abs(te) > 1e-9 else np.nan
        return te, nde, nie, pm
    pt = fit_effects(d, X, Y)
    boots = []
    for _ in range(n_boot):
        try:
            boots.append(fit_effects(
                d.iloc[rng.randint(0, len(d), len(d))].reset_index(drop=True), X, Y))
        except Exception:
            continue
    boots = np.array(boots)
    ci = np.nanpercentile(boots, [2.5, 97.5], axis=0)
    return pt, ci

import os
for med, mtype, lab in ([] if os.environ.get('SKIP_P1') == '1' else
                        [('vaso_new_upgrade', 'bin', '血管活性药新发/升级'),
                         ('log_ned', 'cont', 'NE最大剂量(log1p)')]):
    dsub = dn.copy()
    pt, ci = mediation_effects(dsub, X, Y, med, mtype, n_boot=500)
    print(f"\n=== 中介: {lab} ===")
    print(f"TE  = {pt[0]:+.4f} (95%CI {ci[0,0]:+.4f} ~ {ci[1,0]:+.4f})")
    print(f"NDE = {pt[1]:+.4f} (95%CI {ci[0,1]:+.4f} ~ {ci[1,1]:+.4f})")
    print(f"NIE = {pt[2]:+.4f} (95%CI {ci[0,2]:+.4f} ~ {ci[1,2]:+.4f})")
    print(f"PM  = {pt[3]*100:.1f}% (95%CI {ci[0,3]*100:.1f}% ~ {ci[1,3]*100:.1f}%)")
    pd.DataFrame({'effect': ['TE', 'NDE', 'NIE', 'PM'],
                  'estimate': pt, 'ci_lo': ci[0], 'ci_hi': ci[1]}
                 ).to_csv(f'{D}/results/aim3_mediation_{med}.csv', index=False)
print("\n[done part1]")


# ---------- 2. NE剂量-反应（RCS，SICM组 vs 非SICM组） ----------
def rcs_curve(dsub, label, ax, color):
    ned = dsub['post_max_ned'].fillna(0).values.astype(float)
    yy = dsub['mort_28d'].values.astype(float)
    Xs = dmatrix("cr(ned, df=4)", {'ned': ned}, return_type='dataframe')
    m = sm.Logit(yy, Xs).fit(disp=0)
    grid = np.linspace(0, np.percentile(ned, 99), 100)
    Xg = dmatrix("cr(ned, df=4)", {'ned': grid}, return_type='dataframe')
    pr = m.predict(Xg)
    ax.plot(grid, pr, color=color, label=f"{label} (n={len(dsub)})")
    rows = {'未用NE': yy[ned == 0].mean()}
    if (ned > 0).sum() > 30:
        t = pd.qcut(pd.Series(ned[ned > 0]), 3, duplicates='drop')
        yy_pos = yy[ned > 0]
        for cat in t.cat.categories:
            idx = (t == cat).values
            rows[f'NE {cat}'] = yy_pos[idx].mean()
    return rows

fig, ax = plt.subplots(figsize=(7, 5))
rows_sicm = rcs_curve(dn[dn['sicm'] == 1], '新发SICM', ax, '#d62728')
rows_non = rcs_curve(dn[dn['sicm'] == 0], '非SICM', ax, '#1f77b4')
ax.set_xlabel('Post-onset max norepinephrine dose')
ax.set_ylabel('Predicted 28-day mortality')
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f'{D}/figures/aim3_ne_dose_response.png', dpi=200)
print("\n=== NE剂量三分位死亡率 ===")
print("新发SICM:", {k: round(v, 3) for k, v in rows_sicm.items()})
print("非SICM  :", {k: round(v, 3) for k, v in rows_non.items()})

m_int = sm.Logit(Y, sm.add_constant(pd.DataFrame(
    {'sicm': dn['sicm'], 'log_ned': dn['log_ned'],
     'inter': dn['sicm'] * dn['log_ned']}).join(dn[CONF]).astype(float))).fit(disp=0)
print(f"\n交互 sicm×log(NE): OR={np.exp(m_int.params['inter']):.3f}, P={m_int.pvalues['inter']:.4f}")

# ---------- 3. 表型比较：既存心脏病组中介（方向一致性） ----------
pdco = df[df['has_hf'] == 1].copy().reset_index(drop=True)
pdco['log_ned'] = np.log1p(pdco['post_max_ned'].fillna(0))
pdco['lactate_max_24h'] = pdco['lactate_max_24h'].fillna(pdco['lactate_max_24h'].median())
for c in ['aki_stage_max', 'invasive_vent']:
    pdco[c] = pdco[c].fillna(0)
Xp = pdco['sicm'].values.astype(float); Yp = pdco['mort_28d'].values.astype(float)
pt_p, ci_p = mediation_effects(pdco, Xp, Yp, 'vaso_new_upgrade', 'bin', n_boot=300)
print(f"\n=== 既存心脏病组中介(血管活性药升级): PM={pt_p[3]*100:.1f}% "
      f"(95%CI {ci_p[0,3]*100:.1f}% ~ {ci_p[1,3]*100:.1f}%) ===")
print("\n[done]")
