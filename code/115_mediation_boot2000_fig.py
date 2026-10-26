# 115_mediation_boot2000_fig.py — mediation analysis (2,000 bootstrap replicates, full CI for
# proportion mediated) and Fig. 3 pathway diagram from the same run (figure/table consistency)
# -*- coding: utf-8 -*-
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from decimal import Decimal, ROUND_HALF_UP
from scipy import stats
import statsmodels.api as sm

SEED = 20260101
rng = np.random.RandomState(SEED)
D = '<PROJECT_ROOT>/exec'

co = pd.read_csv(f'{D}/data/cohort_main.csv')
aki = pd.read_csv(f'{D}/data/aki_stage_24h.csv')
st = pd.read_csv(f'{D}/data/cohort_dn_strict24h.csv', usecols=['stay_id', 'vent_24h'])
dn = co[(co['has_hf'] == 0) & (co['congestive_heart_failure'] == 0)].copy().reset_index(drop=True)
dn = dn.merge(aki, on='stay_id', how='left').merge(st, on='stay_id', how='left')
dn['aki_stage_24h'] = dn['aki_stage_24h'].fillna(0).astype(int)
dn['vent_24h'] = dn['vent_24h'].fillna(0).astype(int)
dn['log_ned'] = np.log1p(dn['post_max_ned'].fillna(0))
dn['lact_f'] = dn['lactate_max_24h'].fillna(dn['lactate_max_24h'].median())
CONF = ['age', 'sofa_24h', 'cci', 'lact_f', 'vent_24h', 'aki_stage_24h']

def mediation(d, mediator, mtype, n_boot=2000):
    def fx(dd):
        n = len(dd)
        MC = sm.add_constant(pd.concat([dd['sicm'], dd[CONF]], axis=1).astype(float))
        mfit = (sm.Logit(dd[mediator], MC).fit(disp=0) if mtype == 'bin'
                else sm.OLS(dd[mediator], MC).fit())
        YC = sm.add_constant(pd.DataFrame({'sicm': dd['sicm'], 'M': dd[mediator]}
                                          ).join(dd[CONF]).astype(float))
        yfit = sm.Logit(dd['mort_28d'].astype(float), YC).fit(disp=0)
        def sim_m(xv):
            Xd = pd.DataFrame({'const': 1.0, 'sicm': float(xv)}, index=range(n)).join(dd[CONF].astype(float))
            if mtype == 'bin':
                return (rng.random(n) < mfit.predict(Xd)).astype(float)
            mu = mfit.predict(Xd)
            return np.maximum(mu + rng.normal(0, np.sqrt(mfit.mse_resid), n), 0)
        def py(xv, Mv):
            Yd = pd.DataFrame({'const': 1.0, 'sicm': float(xv), 'M': Mv}).join(dd[CONF].astype(float))
            return yfit.predict(Yd)
        M0, M1 = sim_m(0.0), sim_m(1.0)
        y00, y11, y10 = py(0.0, M0).mean(), py(1.0, M1).mean(), py(1.0, M0).mean()
        te, nde, nie = y11 - y00, y10 - y00, y11 - y10
        return te, nde, nie, (nie / te if abs(te) > 1e-9 else np.nan)
    pt = fx(d)
    boots = []
    for _ in range(n_boot):
        try:
            boots.append(fx(d.iloc[rng.randint(0, len(d), len(d))].reset_index(drop=True)))
        except Exception:
            continue
    ci = np.nanpercentile(np.array(boots), [2.5, 97.5], axis=0)
    return pt, ci

meds = []
canon = []
for med, mtype, lab, short in [('vaso_new_upgrade', 'bin', 'MIMIC: vasopressor initiation',
                                'vasopressor initiation/up-titration'),
                               ('log_ned', 'cont', 'MIMIC: peak norepinephrine (log)',
                                'peak norepinephrine (log1p)')]:
    pt, ci = mediation(dn, med, mtype)
    dfm = pd.DataFrame({'estimate': pt[:3], 'ci_lo': ci[0][:3], 'ci_hi': ci[1][:3]},
                       index=['TE', 'NDE', 'NIE'])
    meds.append((lab, dfm))
    canon.append(dict(mediator=short, TE=pt[0], TE_lo=ci[0][0], TE_hi=ci[1][0],
                      NDE=pt[1], NDE_lo=ci[0][1], NDE_hi=ci[1][1],
                      NIE=pt[2], NIE_lo=ci[0][2], NIE_hi=ci[1][2],
                      PM=pt[3], PM_lo=ci[0][3], PM_hi=ci[1][3]))
    print(f'{short}: TE={pt[0]*100:.1f}pp ({ci[0][0]*100:.1f},{ci[1][0]*100:.1f}) | '
          f'NDE={pt[1]*100:.1f}pp ({ci[0][1]*100:.1f},{ci[1][1]*100:.1f}) | '
          f'NIE={pt[2]*100:.2f}pp ({ci[0][2]*100:.2f},{ci[1][2]*100:.2f}) | '
          f'PM={pt[3]*100:.1f}% ({ci[0][3]*100:.1f},{ci[1][3]*100:.1f})')
pd.DataFrame(canon).to_csv(f'{D}/results/v6_mediation_boot2000.csv', index=False)

ei_m = pd.read_csv(f'{D}/results/aim_eicu_V3_mediation.csv').set_index('effect')
meds.append(('eICU: norepinephrine use (proxy)', ei_m))
print('[mediation done]')

def fmt(v):
    dv = Decimal(str(v * 100))
    if abs(dv) < 1:
        return f"{dv.quantize(Decimal('0.01'), ROUND_HALF_UP):+}"
    return f"{dv.quantize(Decimal('0.1'), ROUND_HALF_UP):+}"

colors = {'TE': '#1f77b4', 'NDE': '#2ca02c', 'NIE': '#d62728'}

def dose_groups(dsub):
    ned_ = dsub['post_max_ned'].fillna(0).values.astype(float)
    yy_ = dsub['mort_28d'].values.astype(float)
    out = [('No norepinephrine', yy_[ned_ == 0])]
    t = pd.qcut(pd.Series(ned_[ned_ > 0]), 3, duplicates='drop')
    for i, cat in enumerate(t.cat.categories):
        out.append((f'T{i+1}', yy_[ned_ > 0][(t == cat).values]))
    return out

fig3, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5.4))
yticks, ylabels, ypos = [], [], 0
for lab, m in meds:
    ylabels.append((ypos + 2.65, lab))          # 组标签置于本组三行之上
    for eff in ['TE', 'NDE', 'NIE']:
        est, lo, hi = m.loc[eff, 'estimate'], m.loc[eff, 'ci_lo'], m.loc[eff, 'ci_hi']
        axA.errorbar(est, ypos, xerr=[[est - lo], [hi - est]], fmt='o',
                     color=colors[eff], capsize=3, ms=5.5)
        axA.text(0.205, ypos, f"{fmt(est)} ({fmt(lo)} to {fmt(hi)})",   # 数值固定右列
                 va='center', ha='left', fontsize=7.5)
        yticks.append((ypos, eff)); ypos += 1
    ypos += 0.8
axA.axvline(0, color='k', ls='--', lw=0.8)
axA.set_yticks([t[0] for t in yticks]); axA.set_yticklabels([t[1] for t in yticks])
for yv, lab in ylabels:
    axA.text(-0.085, yv, lab, fontweight='bold', fontsize=8.5, va='bottom')
axA.set_xlabel('Effect on 28-day mortality (risk difference)')
axA.set_xlim(-0.09, 0.315)
axA.set_ylim(-0.8, 10.9)
axA.legend(handles=[Line2D([0], [0], marker='o', color=colors[k], label=l, ls='')
                    for k, l in [('TE', 'Total effect'), ('NDE', 'Direct effect'),
                                 ('NIE', 'Indirect effect')]],
           loc='upper right', bbox_to_anchor=(0.68, 0.99), fontsize=8)
axA.set_title('A. Mediation effects (95% CI)')
axA.grid(alpha=0.3, axis='x')
for gi, (lab, mask, c) in enumerate([('New-onset SICM', dn['sicm'] == 1, '#d62728'),
                                     ('Reference', dn['sicm'] == 0, '#1f77b4')]):
    for j, (gl, vals) in enumerate(dose_groups(dn[mask])):
        x = j + (0.18 if gi == 0 else -0.18)
        m_ = vals.mean()
        lo, hi = stats.binomtest(int(vals.sum()), len(vals)).proportion_ci(0.95)
        axB.errorbar(x, m_, yerr=[[m_ - lo], [hi - m_]], fmt='o', color=c,
                     capsize=4, ms=7, label=lab if j == 0 else None)
        axB.text(x, hi + 0.02, f"{m_*100:.1f}%", ha='center', fontsize=8, color=c)
axB.set_xticks(range(4))
axB.set_xticklabels(['No NE', 'Tertile 1', 'Tertile 2', 'Tertile 3'], fontsize=9)
axB.set_xlabel('Peak norepinephrine-equivalent dose (µg/kg/min, tertiles among users)')
axB.set_ylabel('Observed 28-day mortality')
axB.set_ylim(0, 0.85)
axB.set_title('B. Observed mortality by dose group (95% CI)')
axB.legend(fontsize=9)
axB.grid(alpha=0.3, axis='y')
fig3.tight_layout()
fig3.savefig(f'{D}/figures/fig3_pathway.png', dpi=200)
plt.close('all')
print('[done: v6_mediation_boot2000.csv + fig3_pathway.png]')
