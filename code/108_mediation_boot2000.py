# 108_mediation_boot2000.py — mediation analysis with 2,000 bootstrap replicates
# -*- coding: utf-8 -*-
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import statsmodels.api as sm

SEED = 20260101
rng = np.random.RandomState(SEED)
D = '<PROJECT_ROOT>/exec'
N_BOOT = 2000

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

def mediation(d, mediator, mtype, n_boot=N_BOOT):
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
    return pt, ci, len(boots)

canon = []
for med, mtype, lab in [('vaso_new_upgrade', 'bin', 'vasopressor initiation/up-titration'),
                        ('log_ned', 'cont', 'peak norepinephrine (log1p)')]:
    pt, ci, nb = mediation(dn, med, mtype)
    canon.append(dict(mediator=lab, TE=pt[0], TE_lo=ci[0][0], TE_hi=ci[1][0],
                      NDE=pt[1], NDE_lo=ci[0][1], NDE_hi=ci[1][1],
                      NIE=pt[2], NIE_lo=ci[0][2], NIE_hi=ci[1][2],
                      PM=pt[3], PM_lo=ci[0][3], PM_hi=ci[1][3], n_boot_ok=nb))
    print(f'{lab}: TE={pt[0]*100:.1f}pp ({ci[0][0]*100:.1f},{ci[1][0]*100:.1f}) | '
          f'NDE={pt[1]*100:.1f}pp ({ci[0][1]*100:.1f},{ci[1][1]*100:.1f}) | '
          f'NIE={pt[2]*100:.2f}pp ({ci[0][2]*100:.2f},{ci[1][2]*100:.2f}) | '
          f'PM={pt[3]*100:.1f}% ({ci[0][3]*100:.1f},{ci[1][3]*100:.1f}) | boots={nb}', flush=True)
pd.DataFrame(canon).to_csv(f'{D}/results/v6_mediation_boot2000.csv', index=False)
print('[done 108: v6_mediation_boot2000.csv]')
