# 102_score_recalibration_figures.py — 补丁：纯取整评分映射 + 63精确重校准 + RCS非线性检验 + NE三分位核验 + 图件重生成(v5)
# -*- coding: utf-8 -*-
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from decimal import Decimal, ROUND_HALF_UP
from scipy import stats
import statsmodels.api as sm
from patsy import dmatrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.calibration import calibration_curve
import xgboost as xgb
import lightgbm as lgb
import shap

SEED = 20260101
rng = np.random.RandomState(SEED)
D = '<PROJECT_ROOT>/exec'

co = pd.read_csv(f'{D}/data/cohort_main.csv')
aki = pd.read_csv(f'{D}/data/aki_stage_24h.csv')
st = pd.read_csv(f'{D}/data/cohort_dn_strict24h.csv',
                 usecols=['stay_id', 'ne_max_24h_strict', 'dobutamine_24h',
                          'milrinone_24h', 'vent_24h'])
co = co.merge(aki, on='stay_id', how='left').merge(st, on='stay_id', how='left')
co['aki_stage_24h'] = co['aki_stage_24h'].fillna(0).astype(int)
dn = co[(co['has_hf'] == 0) & (co['congestive_heart_failure'] == 0)].copy().reset_index(drop=True)
leg = co[co['has_hf'] == 0].copy().reset_index(drop=True)

NUM = ['age','cci','sofa_24h','heart_rate_max','mbp_min','resp_rate_max','spo2_min',
       'temperature_max','lactate_max_24h','wbc_max','platelets_min','creatinine_max',
       'bun_max','bilirubin_total_max','sodium_min','potassium_max','glucose_max',
       'inr_max','ptt_max','bicarbonate_min','aniongap_max','hemoglobin_min',
       'ne_max_24h_strict','aki_stage_24h']
COMORB = ['myocardial_infarct','peripheral_vascular_disease','cerebrovascular_disease',
          'dementia','chronic_pulmonary_disease','rheumatic_disease','mild_liver_disease',
          'diabetes_without_cc','diabetes_with_cc','renal_disease','malignant_cancer',
          'severe_liver_disease','metastatic_solid_tumor','aids']
BIN = ['gender_f','race_black','race_other'] + COMORB + \
      ['vent_24h','dobutamine_24h','milrinone_24h','rrt_firstday','lactate_miss','bili_miss']

def enrich(d):
    d = d.copy()
    d['gender_f'] = (d['gender'] == 'F').astype(int)
    rc = d['race'].astype(str).str.upper()
    d['race_black'] = rc.str.contains('BLACK').astype(int)
    d['race_other'] = (~rc.str.contains('WHITE|BLACK')).astype(int)
    for c in COMORB + ['rrt_firstday']:
        d[c] = d[c].fillna(0).astype(int)
    return d

dn = enrich(dn); leg = enrich(leg)
FEATS = [f for f in (NUM + BIN) if f != 'aids']

def prep(a, b):
    med = a[NUM].median()
    Xa = a[FEATS].copy(); Xb = b[FEATS].copy()
    Xa[NUM] = Xa[NUM].fillna(med); Xb[NUM] = Xb[NUM].fillna(med)
    sc = StandardScaler().fit(Xa[NUM])
    Xa[NUM] = sc.transform(Xa[NUM]); Xb[NUM] = sc.transform(Xb[NUM])
    return Xa, Xb

y = dn['sicm'].values
tr, te = train_test_split(dn.index, test_size=0.3, stratify=y, random_state=SEED)
dtr, dte = dn.loc[tr].copy(), dn.loc[te].copy()
Xtr, Xte = prep(dtr, dte)
ytr, yte = y[tr], y[te]

# 重拟合主模型与比较器（与101一致，供Fig2与评分映射）
las = LogisticRegressionCV(Cs=20, penalty='l1', solver='liblinear', cv=10,
                           scoring='roc_auc', max_iter=3000, random_state=SEED).fit(Xtr, ytr)
p1 = las.predict_proba(Xte)[:, 1]
momed = dtr[['age', 'sofa_24h']].median()
mb = LogisticRegression(max_iter=1000).fit(dtr[['age', 'sofa_24h']].fillna(momed), ytr)
pb = mb.predict_proba(dte[['age', 'sofa_24h']].fillna(momed))[:, 1]
itr, ival = train_test_split(np.arange(len(Xtr)), test_size=0.2, stratify=ytr, random_state=SEED)
mx = xgb.train(dict(objective='binary:logistic', eval_metric='auc', max_depth=3, eta=0.03,
                    subsample=0.8, colsample_bytree=0.8, min_child_weight=20, seed=SEED),
               xgb.DMatrix(Xtr.iloc[itr], label=ytr[itr]), num_boost_round=2000,
               evals=[(xgb.DMatrix(Xtr.iloc[ival], label=ytr[ival]), 'val')],
               early_stopping_rounds=50, verbose_eval=False)
p_val = mx.predict(xgb.DMatrix(Xtr.iloc[ival]), iteration_range=(0, mx.best_iteration + 1))
p_raw = mx.predict(xgb.DMatrix(Xte), iteration_range=(0, mx.best_iteration + 1))
p_xgb = LogisticRegression(max_iter=1000).fit(p_val.reshape(-1, 1), ytr[ival]) \
          .predict_proba(p_raw.reshape(-1, 1))[:, 1]
ml = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.03, max_depth=3, subsample=0.8,
                        colsample_bytree=0.8, min_child_samples=20, random_state=SEED,
                        verbose=-1, n_jobs=-1)
ml.fit(Xtr.iloc[itr], ytr[itr], eval_set=[(Xtr.iloc[ival], ytr[ival])],
       callbacks=[lgb.early_stopping(50, verbose=False)])
p_val_l = ml.predict_proba(Xtr.iloc[ival])[:, 1]
p_raw_l = ml.predict_proba(Xte)[:, 1]
p_lgb = LogisticRegression(max_iter=1000).fit(p_val_l.reshape(-1, 1), ytr[ival]) \
          .predict_proba(p_raw_l.reshape(-1, 1))[:, 1]
print('[models refit]')

# ============ ① 纯取整评分卡映射（points_raw） ============
sc_df = pd.read_csv(f'{D}/results/v5_score_card_full.csv')
sel = [f for f in sc_df.feature if f != '(Intercept)']
beta = sc_df.set_index('feature').loc[sel, 'coef_z']
pts_raw = sc_df.set_index('feature').loc[sel, 'points_raw']
cont = [v for v in sel if v in NUM]
mu = dtr[cont].mean(); sd_ = dtr[cont].std().replace(0, 1)
dtr_z = dtr[sel].astype(float).copy(); dte_z = dte[sel].astype(float).copy()
dtr_z[cont] = (dtr_z[cont].fillna(dtr[cont].median()) - mu) / sd_
dte_z[cont] = (dte_z[cont].fillna(dtr[cont].median()) - mu) / sd_
tot_tr = (dtr_z * pts_raw).sum(axis=1); tot_te = (dte_z * pts_raw).sum(axis=1)
mapfit = sm.Logit(ytr, sm.add_constant(tot_tr.astype(float))).fit(disp=0)
auc_raw = roc_auc_score(yte, tot_te)
print(f'① 纯取整评分: AUC={auc_raw:.3f} (全模型0.731), 总分范围[{int(tot_tr.min())},{int(tot_tr.max())}], '
      f'logit(p)={mapfit.params.iloc[0]:.3f}{mapfit.params.iloc[1]:+.3f}×总分')
pd.DataFrame({'item': ['unit', 'intercept_z', 'map_intercept', 'map_slope',
                       'score_min', 'score_max', 'auc_points'],
              'value': [0.190, -1.609, round(mapfit.params.iloc[0], 3),
                        round(mapfit.params.iloc[1], 3), int(tot_tr.min()), int(tot_tr.max()),
                        round(auc_raw, 3)]}).to_csv(f'{D}/results/v5_score_map_raw.csv', index=False)

# ============ ② 63精确重校准（新OOF, leg队列） ============
oofL = np.zeros(len(leg))
yl = leg['sicm'].values
for tr3, va in StratifiedKFold(5, shuffle=True, random_state=SEED).split(leg, yl):
    Xa, Xb = prep(leg.iloc[tr3], leg.iloc[va])
    m_ = LogisticRegressionCV(Cs=10, penalty='l1', solver='liblinear', cv=5,
                              scoring='roc_auc', max_iter=3000, random_state=SEED).fit(Xa, yl[tr3])
    oofL[va] = m_.predict_proba(Xb)[:, 1]

def ici(yy, pp, n_bins=8):
    pc = calibration_curve(yy, pp, n_bins=n_bins, strategy='quantile')
    return float(np.mean(np.abs(pc[1] - pc[0])))

def cal_metrics(yy, pp):
    pp = np.clip(pp, 1e-6, 1 - 1e-6); lp = np.log(pp / (1 - pp))
    inter = np.asarray(sm.GLM(yy, np.ones((len(yy), 1)), family=sm.families.Binomial(),
                              offset=lp).fit().params)[0]
    prm = np.asarray(sm.Logit(yy, sm.add_constant(lp)).fit(disp=0).params)
    return inter, (prm[1] if len(prm) > 1 else np.nan)

leg['aki_grp'] = leg['aki_stage_max'].fillna(0).astype(int)
leg['sofa_grp'] = (leg['sofa_24h'] >= 8).astype(int)
def subgroup_recal(mask, name):
    idx = np.where(mask.values)[0]
    yy, pp = yl[idx], oofL[idx]
    before = ici(yy, pp)
    pp_rec = np.zeros_like(pp)
    kf = StratifiedKFold(2, shuffle=True, random_state=SEED)
    lp_full = np.log(np.clip(pp, 1e-6, 1 - 1e-6) / (1 - np.clip(pp, 1e-6, 1 - 1e-6)))
    for a, b in kf.split(pp.reshape(-1, 1), yy):
        rec = LogisticRegression(max_iter=1000).fit(lp_full[a].reshape(-1, 1), yy[a])
        pp_rec[b] = rec.predict_proba(lp_full[b].reshape(-1, 1))[:, 1]
    after = ici(yy, pp_rec)
    print(f'② {name}: ICI {before:.3f} -> {after:.3f}')
    return dict(subgroup=name, ici_pre=round(before, 3), ici_post=round(after, 3))
rr = [subgroup_recal(leg['aki_grp'] == 1, 'AKI 1'), subgroup_recal(leg['sofa_grp'] == 1, 'SOFA >=8'),
      subgroup_recal(leg['aki_grp'] == 0, 'AKI 0')]
pd.DataFrame(rr).to_csv(f'{D}/results/v5_recalibration.csv', index=False)

# ============ ③ RCS非线性检验（修正队列, NE剂量→28d死亡） ============
ned = dn['post_max_ned'].fillna(0).values.astype(float)
yy = dn['mort_28d'].values.astype(float)
Xs4 = dmatrix("cr(ned, df=4)", {'ned': ned}, return_type='dataframe')
m_full = sm.Logit(yy, Xs4).fit(disp=0)
m_lin = sm.Logit(yy, sm.add_constant(ned)).fit(disp=0)
lr = 2 * (m_full.llf - m_lin.llf)
p_nl = 1 - stats.chi2.cdf(lr, 3)
print(f'③ RCS(4 knots)非线性LR检验: chi2={lr:.2f}, df=3, P={p_nl:.3f}')
pd.DataFrame({'item': ['rcs_lr_chi2', 'df', 'p_nonlinear'],
              'value': [round(lr, 2), 3, round(p_nl, 3)]}).to_csv(
    f'{D}/results/v5_rcs_nonlinear.csv', index=False)

# ============ ④ NE三分位死亡率核验（修正队列, 应与v4一致） ============
def tert(dsub, lab):
    n_, y_ = dsub['post_max_ned'].fillna(0).values, dsub['mort_28d'].values.astype(float)
    out = {'no NE': y_[n_ == 0].mean()}
    t = pd.qcut(pd.Series(n_[n_ > 0]), 3, duplicates='drop')
    for cat in t.cat.categories:
        out[str(cat)] = y_[n_ > 0][(t == cat).values].mean()
    print(f'④ {lab}: ' + ' | '.join(f'{k}:{v:.3f}' for k, v in out.items()))
tert(dn[dn['sicm'] == 1], 'SICM')
tert(dn[dn['sicm'] == 0], 'reference')

# ============ ⑤ Fig2 模型性能三联图（v5） ============
t2 = pd.read_csv(f'{D}/results/v5_model_fulleval.csv')
auc_of = dict(zip(t2.model, t2.AUC))
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
ax = axes[0]
for name, p, c in [('LASSO-logistic (primary)', p1, '#d62728'),
                   ('XGBoost (Platt)', p_xgb, '#ff7f0e'),
                   ('LightGBM (Platt)', p_lgb, '#2ca02c'),
                   ('Baseline (age+SOFA)', pb, '#7f7f7f')]:
    fpr, tpr, _ = roc_curve(yte, p)
    ax.plot(fpr, tpr, color=c, lw=1.6,
            label=f"{name} (AUC {roc_auc_score(yte, p):.3f})")
ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate')
ax.set_title('A. ROC (test set)'); ax.legend(fontsize=8, loc='lower right'); ax.grid(alpha=0.3)
ax = axes[1]
for name, p, c in [('LASSO-logistic (primary)', p1, '#d62728'),
                   ('XGBoost (Platt)', p_xgb, '#ff7f0e'),
                   ('LightGBM (Platt)', p_lgb, '#2ca02c'),
                   ('Baseline (age+SOFA)', pb, '#7f7f7f')]:
    pc_ = calibration_curve(yte, p, n_bins=10, strategy='quantile')
    ax.plot(pc_[1], pc_[0], 'o-', color=c, lw=1.4, ms=4, label=name)
ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
ax.set_xlabel('Predicted probability'); ax.set_ylabel('Observed frequency')
ax.set_title('B. Calibration (10 quantile bins)'); ax.legend(fontsize=8, loc='upper left')
ax.grid(alpha=0.3)
ax = axes[2]
dca = pd.read_csv(f'{D}/results/v5_dca.csv')
for name, c, ls in [('LASSO lambda.min', '#d62728', '-'), ('Baseline', '#7f7f7f', '-'),
                    ('Treat all', '#1f77b4', '--'), ('Treat none', '#000000', '--')]:
    dd = dca[dca.model == name]
    lab = 'LASSO-logistic (primary)' if name == 'LASSO lambda.min' else name
    ax.plot(dd.threshold, dd.net_benefit, label=lab, color=c, ls=ls, lw=1.4)
ax.set_xlabel('Threshold probability'); ax.set_ylabel('Net benefit')
ax.set_ylim(-0.05, 0.3); ax.set_title('C. Decision-curve analysis')
ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f'{D}/figures/fig2_model_performance.png', dpi=200)
plt.close('all'); print('⑤ fig2 saved')

# ============ ⑥ Fig3 路径双联图（中介森林+剂量分组, v5） ============
def fmt(v):
    dv = Decimal(str(v * 100))
    if abs(dv) < 1:
        return f"{dv.quantize(Decimal('0.01'), ROUND_HALF_UP):+}"
    return f"{dv.quantize(Decimal('0.1'), ROUND_HALF_UP):+}"

# 中介重算（含NIE置信区间, 与101同配置严格CONF）
dn['log_ned'] = np.log1p(dn['post_max_ned'].fillna(0))
dn['lact_f'] = dn['lactate_max_24h'].fillna(dn['lactate_max_24h'].median())
dn['vent_24h'] = dn['vent_24h'].fillna(0).astype(int)
CONF = ['age', 'sofa_24h', 'cci', 'lact_f', 'vent_24h', 'aki_stage_24h']

def mediation(d, mediator, mtype, n_boot=300):
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
for med, mtype, lab in [('vaso_new_upgrade', 'bin', 'MIMIC: vasopressor initiation'),
                        ('log_ned', 'cont', 'MIMIC: peak norepinephrine (log)')]:
    pt, ci = mediation(dn, med, mtype)
    dfm = pd.DataFrame({'estimate': pt[:3], 'ci_lo': ci[0][:3], 'ci_hi': ci[1][:3]},
                       index=['TE', 'NDE', 'NIE'])
    meds.append((lab, dfm))
ei_m = pd.read_csv(f'{D}/results/aim_eicu_V3_mediation.csv').set_index('effect')
meds.append(('eICU: norepinephrine use (proxy)', ei_m))
# 保存规范中介结果（覆盖101版, 正文引用以此为准）
canon = []
for (lab, dfm) in meds[:2]:
    pm_est = dfm.loc['NIE', 'estimate'] / dfm.loc['TE', 'estimate']
    canon.append(dict(mediator=lab, TE=dfm.loc['TE', 'estimate'],
                      TE_lo=dfm.loc['TE', 'ci_lo'], TE_hi=dfm.loc['TE', 'ci_hi'],
                      NDE=dfm.loc['NDE', 'estimate'], NDE_lo=dfm.loc['NDE', 'ci_lo'],
                      NDE_hi=dfm.loc['NDE', 'ci_hi'], NIE=dfm.loc['NIE', 'estimate'],
                      NIE_lo=dfm.loc['NIE', 'ci_lo'], NIE_hi=dfm.loc['NIE', 'ci_hi'],
                      PM=pm_est))
    print(f"规范中介 {lab}: TE={dfm.loc['TE','estimate']*100:.1f}pp "
          f"({dfm.loc['TE','ci_lo']*100:.1f},{dfm.loc['TE','ci_hi']*100:.1f}) "
          f"NDE={dfm.loc['NDE','estimate']*100:.1f}pp "
          f"({dfm.loc['NDE','ci_lo']*100:.1f},{dfm.loc['NDE','ci_hi']*100:.1f}) "
          f"PM={pm_est*100:.1f}%")
pd.DataFrame(canon).to_csv(f'{D}/results/v5_mediation.csv', index=False)
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
    ylabels.append((ypos + 1.5, lab))
    for eff in ['TE', 'NDE', 'NIE']:
        est, lo, hi = m.loc[eff, 'estimate'], m.loc[eff, 'ci_lo'], m.loc[eff, 'ci_hi']
        axA.errorbar(est, ypos, xerr=[[est - lo], [hi - est]], fmt='o',
                     color=colors[eff], capsize=3, ms=5.5)
        axA.text(hi + 0.005, ypos, f"{fmt(est)} ({fmt(lo)} to {fmt(hi)})",
                 va='center', fontsize=7.5)
        yticks.append((ypos, eff)); ypos += 1
    ypos += 1.5
axA.axvline(0, color='k', ls='--', lw=0.8)
axA.set_yticks([t[0] for t in yticks]); axA.set_yticklabels([t[1] for t in yticks])
for yv, lab in ylabels:
    axA.text(-0.09, yv, lab, fontweight='bold', fontsize=8.5, va='bottom')
axA.set_xlabel('Effect on 28-day mortality (risk difference)')
axA.set_xlim(-0.09, 0.30)
axA.legend(handles=[Line2D([0], [0], marker='o', color=colors[k], label=l, ls='')
                    for k, l in [('TE', 'Total effect'), ('NDE', 'Direct effect'),
                                 ('NIE', 'Indirect effect')]],
           loc='upper left', bbox_to_anchor=(0.0, 1.0), fontsize=8)
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
plt.close('all'); print('⑥ fig3 saved')

# ============ ⑦ Fig4 SHAP（v5特征, 可读标签） ============
FEAT_EN = {'myocardial_infarct': 'Prior myocardial infarction',
           'ptt_max': 'PTT max', 'glucose_max': 'Glucose max',
           'heart_rate_max': 'Heart rate max', 'temperature_max': 'Temperature max',
           'hemoglobin_min': 'Haemoglobin min', 'platelets_min': 'Platelets min',
           'sodium_min': 'Sodium min', 'aniongap_max': 'Anion gap max',
           'resp_rate_max': 'Respiratory rate max', 'age': 'Age',
           'wbc_max': 'WBC max', 'potassium_max': 'Potassium max',
           'inr_max': 'INR max', 'bilirubin_total_max': 'Bilirubin max',
           'creatinine_max': 'Creatinine max', 'bun_max': 'BUN max',
           'lactate_max_24h': 'Lactate max', 'spo2_min': 'SpO2 min',
           'mbp_min': 'MAP min', 'sofa_24h': 'SOFA score', 'cci': 'Charlson index',
           'ne_max_24h_strict': 'Peak norepinephrine (24 h)',
           'aki_stage_24h': 'AKI stage (0–24 h)', 'bicarbonate_min': 'Bicarbonate min',
           'dobutamine_24h': 'Dobutamine (24 h)', 'milrinone_24h': 'Milrinone (24 h)',
           'vent_24h': 'Invasive ventilation (24 h)', 'rrt_firstday': 'RRT (day 1)',
           'gender_f': 'Female sex', 'race_black': 'Black race', 'race_other': 'Other race',
           'lactate_miss': 'Lactate missing', 'bili_miss': 'Bilirubin missing',
           'peripheral_vascular_disease': 'Peripheral vascular disease',
           'cerebrovascular_disease': 'Cerebrovascular disease', 'dementia': 'Dementia',
           'chronic_pulmonary_disease': 'Chronic pulmonary disease',
           'rheumatic_disease': 'Rheumatic disease', 'mild_liver_disease': 'Mild liver disease',
           'diabetes_without_cc': 'Diabetes', 'diabetes_with_cc': 'Diabetes with complications',
           'renal_disease': 'Renal disease', 'malignant_cancer': 'Malignancy',
           'severe_liver_disease': 'Severe liver disease',
           'metastatic_solid_tumor': 'Metastatic tumour', 'aids': 'AIDS'}
X0 = dn[FEATS].fillna(dn[FEATS].median())
sc0 = StandardScaler().fit(X0[NUM])
Xs0 = X0.copy(); Xs0[NUM] = sc0.transform(Xs0[NUM])
mx2 = xgb.train(dict(objective='binary:logistic', eval_metric='auc', max_depth=3, eta=0.05,
                     subsample=0.8, colsample_bytree=0.8, min_child_weight=20, seed=SEED),
                xgb.DMatrix(Xs0, label=y), num_boost_round=400, verbose_eval=False)
sv = shap.TreeExplainer(mx2).shap_values(Xs0)
labels = [FEAT_EN.get(f, f) for f in FEATS]
plt.figure(figsize=(8, 6))
shap.summary_plot(sv, Xs0, feature_names=labels, show=False, max_display=15)
plt.tight_layout()
plt.savefig(f'{D}/figures/fig4_shap_corrected.png', dpi=200, bbox_inches='tight')
plt.close('all')
imp = pd.DataFrame({'feature': FEATS, 'mean_abs_shap': np.abs(sv).mean(axis=0)}
                   ).sort_values('mean_abs_shap', ascending=False)
imp.to_csv(f'{D}/results/v5_shap_importance.csv', index=False)
print('⑦ fig4 saved; SHAP Top10:'); print(imp.head(10).to_string(index=False))

# ============ ⑧ FigS1 亚组森林图（v5） ============
sg = pd.read_csv(f'{D}/results/v5_subgroup_heterogeneity.csv').dropna(subset=['AUC'])
sg['lab'] = sg['dim'] + ': ' + sg['subgroup']
sg = sg.iloc[::-1]
fig, ax = plt.subplots(figsize=(8, 6))
ax.errorbar(sg.AUC, range(len(sg)),
            xerr=[sg.AUC - sg.lo, sg.hi - sg.AUC], fmt='o', color='#1f77b4',
            capsize=3, ms=5)
oof_overall = pd.read_csv(f'{D}/results/v5_weighted_auc.csv').iloc[0, 1]
ax.axvline(oof_overall, color='r', ls='--', lw=1,
           label=f'Overall OOF AUC = {oof_overall:.3f}')
ax.set_yticks(range(len(sg))); ax.set_yticklabels(sg.lab, fontsize=8)
ax.set_xlabel('Out-of-fold AUC (95% CI)')
ax.set_xlim(0.4, 1.0)
ax.legend(fontsize=9, loc='lower right'); ax.grid(alpha=0.3, axis='x')
fig.tight_layout()
fig.savefig(f'{D}/figures/figS_subgroup_forest.png', dpi=200)
plt.close('all'); print('⑧ figS1 saved')
print('[done 102]')

