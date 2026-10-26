# 101_full_recompute_strict24h.py — AKI分期泄漏修复后的全链路重算（v5系列输出）
# 背景: aki_stage_max 来自 kdigo_stages 全病程MAX（58%患者分期在24h后升高）→ 预测模型泄漏+关联分析暴露后调整
# 修复: 模型特征与调整集统一换用严格0-24h分期 aki_stage_24h（kdigo_stages按charttime限窗）；
#       关联/中介调整集 invasive_vent(whole-stay) 同步换为 vent_24h
# 输出: v5_* 系列CSV，不覆盖v4原文件
# -*- coding: utf-8 -*-
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, psycopg2
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve
import xgboost as xgb
import lightgbm as lgb

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
print(f'修正队列 n={len(dn)}, 事件={dn.sicm.sum()}; 旧队列 n={len(leg)}, 事件={leg.sicm.sum()}')

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

def las_fit20(X, yy):
    return LogisticRegressionCV(Cs=20, penalty='l1', solver='liblinear', cv=10,
                                scoring='roc_auc', max_iter=3000, random_state=SEED).fit(X, yy)

def las_fit10(X, yy):
    return LogisticRegressionCV(Cs=10, penalty='l1', solver='liblinear', cv=5,
                                scoring='roc_auc', max_iter=3000, random_state=SEED).fit(X, yy)

def cal_stats(yy, p):
    p = np.clip(p, 1e-6, 1 - 1e-6); lp = np.log(p / (1 - p))
    inter = np.asarray(sm.GLM(yy, np.ones((len(yy), 1)), family=sm.families.Binomial(),
                              offset=lp).fit().params)[0]
    prm = np.asarray(sm.Logit(yy, sm.add_constant(lp)).fit(disp=0).params)
    pc = calibration_curve(yy, p, n_bins=10, strategy='quantile')
    return inter, (prm[1] if len(prm) > 1 else np.nan), np.mean(np.abs(pc[1] - pc[0]))

def boot_ci(yy, pp, n=1000):
    a = []
    for _ in range(n):
        i = rng.randint(0, len(yy), len(yy))
        if len(np.unique(yy[i])) > 1:
            a.append(roc_auc_score(yy[i], pp[i]))
    return np.percentile(a, [2.5, 97.5])

y = dn['sicm'].values
tr, te = train_test_split(dn.index, test_size=0.3, stratify=y, random_state=SEED)
dtr, dte = dn.loc[tr].copy(), dn.loc[te].copy()
Xtr, Xte = prep(dtr, dte)
ytr, yte = y[tr], y[te]
print(f'测试集 n={len(te)}, 事件={yte.sum()}')


# ============ Part 1: 主模型链（Table 2） ============
rows = []
momed = dtr[['age', 'sofa_24h']].median()
mb = LogisticRegression(max_iter=1000).fit(dtr[['age', 'sofa_24h']].fillna(momed), ytr)
pb = mb.predict_proba(dte[['age', 'sofa_24h']].fillna(momed))[:, 1]
las = las_fit20(Xtr, ytr)
p1 = las.predict_proba(Xte)[:, 1]
means = las.scores_[1].mean(axis=0); ses = las.scores_[1].std(axis=0) / np.sqrt(10)
c1se = las.Cs_[np.where(means >= means.max() - ses[means.argmax()])[0][0]]
print(f'lambda.min C={las.C_[0]:.5f} vs 1-SE C={c1se:.5f} (相同={np.isclose(las.C_[0], c1se)})')
las1 = LogisticRegression(C=c1se, penalty='l1', solver='liblinear',
                          max_iter=3000, random_state=SEED).fit(Xtr, ytr)
p2 = las1.predict_proba(Xte)[:, 1]
sel1 = [f for f, c in zip(FEATS, las1.coef_[0]) if abs(c) > 1e-8]
print(f'1-SE变量数: {len(sel1)}: {sel1}')

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

coef_lm = pd.Series(las.coef_[0], index=FEATS)
sel_lm = list(coef_lm[coef_lm.abs() > 1e-8].index)
print(f'lambda.min变量数: {len(sel_lm)}: {sel_lm}')

for name, p in [('Baseline (age+SOFA)', pb), ('LASSO-logistic (lambda.min; primary)', p1),
                ('LASSO-logistic (1-SE, sparse)', p2),
                ('XGBoost (Platt-calibrated)', p_xgb), ('LightGBM (Platt-calibrated)', p_lgb)]:
    inter, slope, ici = cal_stats(yte, p)
    ci = boot_ci(yte, p)
    rows.append(dict(model=name, AUC=round(roc_auc_score(yte, p), 3),
                     PR_AUC=round(average_precision_score(yte, p), 3),
                     AUC_lo=round(ci[0], 3), AUC_hi=round(ci[1], 3),
                     Brier=round(brier_score_loss(yte, p), 3),
                     Cal_intercept=round(inter, 3), Cal_slope=round(slope, 3),
                     ICI=round(ici, 3)))
t2 = pd.DataFrame(rows)
print('\n=== Part1 Table 2 (v5) ==='); print(t2.to_string(index=False))
t2.to_csv(f'{D}/results/v5_model_fulleval.csv', index=False)

ths = np.linspace(0.05, 0.5, 46); prev = yte.mean()
dca = []
for t_ in ths:
    for name, p in [('LASSO lambda.min', p1), ('Baseline', pb)]:
        pred = (p >= t_).astype(int)
        tp = ((pred == 1) & (yte == 1)).sum(); fp = ((pred == 1) & (yte == 0)).sum()
        dca.append(dict(threshold=t_, model=name,
                        net_benefit=tp / len(yte) - fp / len(yte) * t_ / (1 - t_)))
dca = pd.DataFrame(dca)
dca = pd.concat([dca, pd.DataFrame({'threshold': ths, 'model': 'Treat all',
                                    'net_benefit': prev - (1 - prev) * ths / (1 - ths)}),
                 pd.DataFrame({'threshold': ths, 'model': 'Treat none', 'net_benefit': 0.0})])
dca.to_csv(f'{D}/results/v5_dca.csv', index=False)
las_d = dca[dca.model == 'LASSO lambda.min'].set_index('threshold')['net_benefit']
best = np.maximum(dca[dca.model == 'Treat all'].set_index('threshold')['net_benefit'], 0.0)
pos = las_d.index[las_d.values > best.values]
print(f'DCA净获益区间: {pos.min():.3f}-{pos.max():.3f}')

# ============ Part 2: OOF（与主模型同规格：Cs=20, 内层10折） ============
oof = np.zeros(len(dn))
for tr2, va in StratifiedKFold(5, shuffle=True, random_state=SEED).split(dn, y):
    Xa, Xb = prep(dn.iloc[tr2], dn.iloc[va])
    oof[va] = las_fit20(Xa, y[tr2]).predict_proba(Xb)[:, 1]
oof_auc = roc_auc_score(y, oof)
print(f'\n=== Part2 OOF AUC = {oof_auc:.3f} (v4: 0.694) ===')
pd.DataFrame({'metric': ['OOF_AUC'], 'value': [round(oof_auc, 3)]}).to_csv(
    f'{D}/results/v5_oof.csv', index=False)

# ============ Part 3: 系数表 + 评分卡（97③④流程） ============
coef = pd.Series(las.coef_[0], index=FEATS)
coef = coef[coef.abs() > 1e-8].sort_values(key=abs, ascending=False)
top = pd.DataFrame({'feature': coef.index, 'std_coef': coef.values.round(3)})
top.to_csv(f'{D}/results/v5_lasso_std_coefficients.csv', index=False)
print('\n=== Part3a 惩罚后标准化系数Top10 ==='); print(top.head(10).to_string(index=False))

m_uni = sm.Logit(y, sm.add_constant(dn['myocardial_infarct'].astype(float)).values).fit(disp=0)
ci_u = np.exp(m_uni.conf_int())
print(f'MI单变量OR={np.exp(m_uni.params[1]):.2f} ({ci_u[1,0]:.2f}-{ci_u[1,1]:.2f})')

sel = list(top['feature'])
cont = [v for v in sel if v in NUM]
mu = dtr[cont].mean(); sd_ = dtr[cont].std().replace(0, 1)
dtr_z = dtr[sel].astype(float).copy()
dtr_z[cont] = dtr_z[cont].fillna(dtr[cont].median())
dtr_z[cont] = (dtr_z[cont] - mu) / sd_
mm = sm.Logit(ytr, sm.add_constant(dtr_z)).fit(disp=0)
beta = mm.params.drop('const')
unit = np.median(np.abs(beta))
pts_raw = (beta / unit).round().astype(int)
pts = pts_raw.copy(); pts[pts == 0] = np.sign(beta[pts == 0]).astype(int)
dte_z = dte[sel].astype(float).copy()
dte_z[cont] = dte_z[cont].fillna(dtr[cont].median())
dte_z[cont] = (dte_z[cont] - mu) / sd_
auc_score = roc_auc_score(yte, (dte_z * pts).sum(axis=1))
auc_score_raw = roc_auc_score(yte, (dte_z * pts_raw).sum(axis=1))
print(f'\n=== Part3b 整数评分卡 AUC={auc_score:.3f} (v4: 0.705; 1-SE模型 {roc_auc_score(yte, p2):.3f})')
print(f'    纯四舍五入(不强制符号) AUC={auc_score_raw:.3f}; unit=median|beta|={unit:.3f}')
sc_rows = []
for f in sel:
    sc_rows.append(dict(feature=f, mean=round(mu[f], 3) if f in cont else np.nan,
                        sd=round(sd_[f], 3) if f in cont else np.nan,
                        coef_z=round(beta[f], 3), points=int(pts[f]),
                        points_raw=int(pts_raw[f])))
sc_df = pd.DataFrame(sc_rows)
sc_df.loc[len(sc_df)] = dict(feature='(Intercept)', mean=np.nan, sd=np.nan,
                             coef_z=round(mm.params['const'], 3), points=np.nan, points_raw=np.nan)
sc_df.to_csv(f'{D}/results/v5_score_card_full.csv', index=False)
# 总分→概率映射（训练集上 outcome ~ total_points 逻辑回归）
tot_tr = (dtr_z * pts).sum(axis=1)
mapfit = sm.Logit(ytr, sm.add_constant(tot_tr.astype(float))).fit(disp=0)
print(f'    总分→概率: logit(p)={mapfit.params.iloc[0]:.3f}+{mapfit.params.iloc[1]:.3f}×总分; '
      f'总分范围[{int(tot_tr.min())},{int(tot_tr.max())}]')
pd.DataFrame({'item': ['unit', 'intercept_z', 'map_intercept', 'map_slope',
                       'score_min', 'score_max', 'auc_points', 'auc_points_raw'],
              'value': [round(unit, 4), round(mm.params['const'], 3),
                        round(mapfit.params.iloc[0], 3), round(mapfit.params.iloc[1], 3),
                        int(tot_tr.min()), int(tot_tr.max()),
                        round(auc_score, 3), round(auc_score_raw, 3)]}).to_csv(
    f'{D}/results/v5_score_map.csv', index=False)

# ============ Part 4: 关联分析链（严格CONF: vent_24h + aki_stage_24h） ============
CONF = ['age', 'sofa_24h', 'cci', 'lact_f', 'vent_24h', 'aki_stage_24h']
for d in (dn, leg):
    d['lact_f'] = d['lactate_max_24h'].fillna(d['lactate_max_24h'].median())
    d['vent_24h'] = d['vent_24h'].fillna(0).astype(int)

def adj_or(dsub, xcol):
    X = sm.add_constant(pd.concat([dsub[xcol], dsub[CONF]], axis=1).astype(float))
    m = sm.Logit(dsub['mort_28d'].astype(float), X).fit(disp=0)
    ci = np.exp(m.conf_int().loc[xcol])
    return np.exp(m.params[xcol]), ci[0], ci[1], m.pvalues[xcol]

o, lo, hi, pv = adj_or(dn, 'sicm')
print(f'\n=== Part4a 修正队列调整OR(严格CONF) = {o:.3f} ({lo:.3f}-{hi:.3f}), P={pv:.4f} (v4: 1.659 (1.172-2.348), P=0.004)')

# IPW（84流程：倾向模型不变，结局模型换严格CONF）
conn = psycopg2.connect(dbname='mimic', host='127.0.0.1', port=5431,
                        user='postgres', password='<SET-IN-ENVIRONMENT>')
full = pd.read_sql("""SELECT stay_id, age, gender, sofa_24h, heart_rate_max, mbp_min,
                      resp_rate_max, spo2_min, lactate_max_24h, wbc_max, creatinine_max,
                      invasive_vent, cci, evaluable
                      FROM mimiciv_echo.sicm_analysis_dataset
                      WHERE age>=18 AND age<=90 AND has_acs=0""", conn)
conn.close()
PSF = ['age','sofa_24h','heart_rate_max','mbp_min','resp_rate_max','spo2_min',
       'lactate_max_24h','wbc_max','creatinine_max','cci']
Xps = full[PSF].copy()
for c in PSF:
    Xps[c] = Xps[c].fillna(Xps[c].median())
Xps['invasive_vent'] = full['invasive_vent'].fillna(0)
Xps['gender_f'] = (full['gender'] == 'F').astype(int)
scp = StandardScaler().fit(Xps)
pm = LogisticRegression(max_iter=1500).fit(scp.transform(Xps), full['evaluable'])
ps = np.clip(pm.predict_proba(scp.transform(Xps))[:, 1], 0.02, 0.98)
prevf = full['evaluable'].mean()
w = np.where(full['evaluable'] == 1, prevf / ps, 1.0)
we = w[full['evaluable'] == 1]
lo_w, hi_w = np.percentile(we, [1, 99])
w[full['evaluable'] == 1] = np.clip(we, lo_w, hi_w)
full['sw'] = w
print(f'PS模型AUC={roc_auc_score(full["evaluable"], ps):.3f} (v4: 0.705)')

def w_or(dsub, xcol, lab):
    d2 = dsub.merge(full[['stay_id', 'sw']], on='stay_id', how='left')
    d2 = d2.dropna(subset=['sw', 'mort_28d', xcol] + CONF)
    d2 = d2[d2['sw'] > 0]
    X = sm.add_constant(pd.concat([d2[xcol], d2[CONF]], axis=1).astype(float))
    mw = sm.GLM(d2['mort_28d'].astype(float), X, family=sm.families.Binomial(),
                freq_weights=d2['sw']).fit(cov_type='HC1')
    cw = np.exp(mw.conf_int().loc[xcol])
    print(f'{lab}: IPW OR={np.exp(mw.params[xcol]):.3f} ({cw[0]:.3f}-{cw[1]:.3f}), P={mw.pvalues[xcol]:.4f}')
    return dict(exposure=lab, OR_iptw=round(np.exp(mw.params[xcol]), 3),
                lo=round(cw[0], 3), hi=round(cw[1], 3), p=float(f'{mw.pvalues[xcol]:.2e}'))

r_ipw = [w_or(dn, 'sicm', '修正新发SICM(主)'), w_or(dn, 'sicm_defB', '失代偿定义B')]
pd.DataFrame(r_ipw).to_csv(f'{D}/results/v5_iptw.csv', index=False)

# 旧口径队列（Table S3左列, 严格CONF）
o2, lo2, hi2, pv2 = adj_or(leg, 'sicm')
print(f'旧口径调整OR = {o2:.3f} ({lo2:.3f}-{hi2:.3f}), P={pv2:.3f} (v4: 1.162 (0.919-1.470), P=0.21)')
r_leg = [w_or(leg, 'sicm', '旧口径SICM')]
print(f'   (v4: IPW 1.010 (0.786-1.297), P=0.94)')

# 敏感性表型OR（97②流程, 严格CONF）
rows = []
for outc, lab in [('sicm_ef40', 'LVEF <40%'), ('sicm_drop15', 'LVEF drop ≥15 pp'),
                  ('sicm_img', 'Imaging-only'), ('sicm_defB', 'Decompensated')]:
    n_ev = int(dn[outc].sum())
    if n_ev < 30:
        rows.append(dict(phenotype=lab, events=n_ev)); continue
    oo, llo, hhi, ppv = adj_or(dn, outc)
    rows.append(dict(phenotype=lab, events=n_ev, OR=round(oo, 3), lo=round(llo, 3),
                     hi=round(hhi, 3), p=float(f'{ppv:.2e}')))
sens = pd.DataFrame(rows)
print('\n=== Part4d 敏感性表型OR(严格CONF) ==='); print(sens.to_string(index=False))
print('(v4: 2.17 (1.41-3.37); 1.69 (1.19-2.40); 1.67 (1.18-2.36); 1.58 (1.00-2.49))')
sens.to_csv(f'{D}/results/v5_sensitivity_phenotypes_mortality.csv', index=False)

pd.DataFrame({'item': ['adj_OR', 'adj_lo', 'adj_hi', 'adj_p',
                       'leg_adj_OR', 'leg_adj_lo', 'leg_adj_hi', 'leg_adj_p'],
              'value': [round(o, 3), round(lo, 3), round(hi, 3), float(f'{pv:.2e}'),
                        round(o2, 3), round(lo2, 3), round(hi2, 3), float(f'{pv2:.2e}')]}
             ).to_csv(f'{D}/results/v5_adj_or.csv', index=False)

# ============ Part 5: 中介分析（85流程, 严格CONF） ============
dn['log_ned'] = np.log1p(dn['post_max_ned'].fillna(0))

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

print('\n=== Part5 中介(严格CONF) ===')
med_rows = []
for med, mtype, lab in [('vaso_new_upgrade', 'bin', 'vasopressor initiation/up-titration'),
                        ('log_ned', 'cont', 'peak norepinephrine (log1p)')]:
    pt, ci = mediation(dn, med, mtype)
    print(f'{lab}: TE={pt[0]*100:.1f}pp ({ci[0][0]*100:.1f},{ci[1][0]*100:.1f}) | '
          f'NDE={pt[1]*100:.1f}pp ({ci[0][1]*100:.1f},{ci[1][1]*100:.1f}) | '
          f'PM={pt[3]*100:.1f}% ({ci[0][3]*100:.1f},{ci[1][3]*100:.1f})')
    med_rows.append(dict(mediator=lab, TE=pt[0], NDE=pt[1], NIE=pt[2], PM=pt[3],
                         TE_lo=ci[0][0], TE_hi=ci[1][0], NDE_lo=ci[0][1], NDE_hi=ci[1][1],
                         PM_lo=ci[0][3], PM_hi=ci[1][3]))
pd.DataFrame(med_rows).to_csv(f'{D}/results/v5_mediation.csv', index=False)
print('(v4: TE +10.0 (3.8-17.0); NDE +10.3 (4.1-17.4), PM -3.1% (-15.9,6.0); '
      'log ned: NDE +9.6 (3.0-16.4), PM 4.0% (-2.9,15.5))')

# ============ Part 6: landmark-24h（97①流程, 新冻结模型） ============
# [v6修正 2026-10-20] 原派生 (t_ef50 NaN | t_ef50>24) 有两处缺陷:
#  (i) 13例仅靠下降标准入组的SICM患者无t_ef50时间戳, 被错误计入"24h无SICM"人群;
#  (ii) 未要求界标时点存活(与方法学描述"alive and without SICM at 24 h"不符).
# 修正后: 剔除24h前发病(142例)+无时间戳的下降标准病例(13例)+24h前死亡(24例) -> n=610, 事件60.
cr = pd.read_csv(f'{D}/data/comprisk_corrected.csv')[['stay_id', 't_ef50', 't_death']]
dn_lm = dn.merge(cr, on='stay_id', how='left')
_early = (dn_lm['sicm'] == 1) & (dn_lm['t_ef50'] <= 24)
_decline_only = (dn_lm['sicm'] == 1) & (dn_lm['t_ef50'].isna())
_dead24 = dn_lm['t_death'].fillna(np.inf) <= 24
lm = dn_lm[~_early & ~_decline_only & ~_dead24].copy().reset_index(drop=True)
ylm = ((lm['sicm'] == 1) & (lm['t_ef50'] > 24)).astype(int).values
print(f'\n=== Part6 landmark-24h n={len(lm)}, 24h后新发={ylm.sum()} ({ylm.mean()*100:.1f}%) (v4: n=644, 60) ')
tr2, te2 = train_test_split(np.arange(len(lm)), test_size=0.3, stratify=ylm, random_state=SEED)
Xa, Xb = prep(lm.iloc[tr2], lm.iloc[te2])
p_trans = las.predict_proba(prep(lm.iloc[tr2], lm.iloc[te2])[1])[:, 1]
auc_trans = roc_auc_score(ylm[te2], p_trans); ci_trans = boot_ci(ylm[te2], p_trans)
m_lm = las_fit20(Xa, ylm[tr2])
p_lm = m_lm.predict_proba(Xb)[:, 1]
auc_lm = roc_auc_score(ylm[te2], p_lm); ci_lm = boot_ci(ylm[te2], p_lm)
print(f'   运输(冻结模型): AUC={auc_trans:.3f} ({ci_trans[0]:.3f}-{ci_trans[1]:.3f}) (v4: 0.698 (0.536-0.852))')
print(f'   landmark内重训: AUC={auc_lm:.3f} ({ci_lm[0]:.3f}-{ci_lm[1]:.3f}) (v4: 0.492 (0.357-0.636))')
pd.DataFrame({'analysis': ['landmark_cohort_n', 'events_after_24h', 'event_rate',
                           'transport_AUC', 'transport_lo', 'transport_hi',
                           'refit_AUC', 'refit_lo', 'refit_hi'],
              'value': [len(lm), int(ylm.sum()), round(ylm.mean(), 3),
                        round(auc_trans, 3), round(ci_trans[0], 3), round(ci_trans[1], 3),
                        round(auc_lm, 3), round(ci_lm[0], 3), round(ci_lm[1], 3)]}
             ).to_csv(f'{D}/results/v6_landmark24h_corrected.csv', index=False)

# ============ Part 7: HF不限队列稳健性电池（n=1504, 严格特征+aki24h） ============
leg['age_grp'] = pd.cut(leg['age'], [0, 65, 75, 200], labels=['<65', '65-74', '>=75'])
leg['sofa_grp'] = (leg['sofa_24h'] >= 8).astype(int)
leg['shock_24h'] = (leg['post_max_ned'] > 0).astype(int)
leg['aki_grp'] = leg['aki_stage_max'].fillna(0).astype(int)
leg['invasive_vent'] = leg['invasive_vent'].fillna(0).astype(int)
yl = leg['sicm'].values

oofL = np.zeros(len(leg))
for tr3, va in StratifiedKFold(5, shuffle=True, random_state=SEED).split(leg, yl):
    Xa, Xb = prep(leg.iloc[tr3], leg.iloc[va])
    oofL[va] = las_fit10(Xa, yl[tr3]).predict_proba(Xb)[:, 1]
oofL_auc = roc_auc_score(yl, oofL)
print(f'\n=== Part7 HF不限队列OOF AUC = {oofL_auc:.3f} (v4/S2: 0.743) ===')

def cal_slope(yy, pp):
    pp = np.clip(pp, 1e-6, 1 - 1e-6); lp = np.log(pp / (1 - pp))
    try:
        return sm.Logit(yy, sm.add_constant(lp)).fit(disp=0).params[1]
    except Exception:
        return np.nan

def boot_ci_auc(yy, pp, n=500):
    a = []
    for _ in range(n):
        i = rng.randint(0, len(yy), len(yy))
        if len(np.unique(yy[i])) < 2: continue
        a.append(roc_auc_score(yy[i], pp[i]))
    return np.percentile(a, [2.5, 97.5]) if a else (np.nan, np.nan)

subgroups = {
    'Age': [('Age <65', leg['age_grp'] == '<65'), ('Age 65-74', leg['age_grp'] == '65-74'),
            ('Age >=75', leg['age_grp'] == '>=75')],
    'SOFA': [('SOFA <8', leg['sofa_grp'] == 0), ('SOFA >=8', leg['sofa_grp'] == 1)],
    'Shock (24h NE)': [('No NE', leg['shock_24h'] == 0), ('NE used', leg['shock_24h'] == 1)],
    'AKI': [(f'AKI {k}', leg['aki_grp'] == k) for k in [0, 1, 2, 3]],
    'Ventilation': [('Not ventilated', leg['invasive_vent'] == 0),
                    ('Ventilated', leg['invasive_vent'] == 1)],
}
sub_rows = []
for dim, lst in subgroups.items():
    for lab, mask in lst:
        m = mask.values
        yy, pp = yl[m], oofL[m]
        if m.sum() < 60 or yy.sum() < 20:
            sub_rows.append(dict(dim=dim, subgroup=lab, n=int(m.sum()), events=int(yy.sum())))
            continue
        ci = boot_ci_auc(yy, pp)
        diffs = []
        for _ in range(500):
            i = rng.randint(0, len(yy), len(yy)); j = rng.randint(0, len(yl), len(yl))
            if len(np.unique(yy[i])) < 2 or len(np.unique(yl[j])) < 2: continue
            diffs.append(roc_auc_score(yy[i], pp[i]) - roc_auc_score(yl[j], oofL[j]))
        diffs = np.array(diffs)
        psub = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
        sub_rows.append(dict(dim=dim, subgroup=lab, n=int(m.sum()), events=int(yy.sum()),
                             AUC=round(roc_auc_score(yy, pp), 3), lo=round(ci[0], 3),
                             hi=round(ci[1], 3), cal_slope=round(cal_slope(yy, pp), 3),
                             p_vs_overall=round(psub, 4)))
sub = pd.DataFrame(sub_rows)
pv_s = sub['p_vs_overall'].dropna().sort_values()
q = pv_s * len(pv_s) / (np.arange(len(pv_s)) + 1)
q = q[::-1].cummin()[::-1]
sub.loc[pv_s.index, 'q_FDR'] = q.values.round(4)
print('=== 亚组（FDR<0.05数:', int((sub.q_FDR < 0.05).sum()), '）===')
print(sub.to_string(index=False))
sub.to_csv(f'{D}/results/v5_subgroup_heterogeneity.csv', index=False)

era_rows = []
for e in sorted(leg['anchor_year_group'].unique()):
    mtr = (leg['anchor_year_group'] != e).values; mva = ~mtr
    if mva.sum() < 50 or yl[mva].sum() < 10:
        era_rows.append(dict(era=e, n=int(mva.sum()), events=int(yl[mva].sum())))
        continue
    Xa, Xb = prep(leg[mtr], leg[mva])
    m_ = las_fit10(Xa, yl[mtr])
    era_rows.append(dict(era=e, n=int(mva.sum()), events=int(yl[mva].sum()),
                         AUC=round(roc_auc_score(yl[mva], m_.predict_proba(Xb)[:, 1]), 3)))
era_tab = pd.DataFrame(era_rows)
print('=== 年代留一法 ==='); print(era_tab.to_string(index=False))
era_tab.to_csv(f'{D}/results/v5_temporal_era_cv.csv', index=False)

sens_rows = []
for outc, lab in [('sicm_ef40', 'LVEF <40%'), ('sicm_drop15', 'LVEF drop ≥15 pp'),
                  ('sicm_img', 'Imaging-only'), ('sicm_defB', 'Decompensated')]:
    yy = leg[outc].values
    tr4, te4 = train_test_split(np.arange(len(leg)), test_size=0.3, stratify=yy, random_state=SEED)
    Xa, Xb = prep(leg.iloc[tr4], leg.iloc[te4])
    m_ = las_fit10(Xa, yy[tr4])
    pte = m_.predict_proba(Xb)[:, 1]
    sens_rows.append(dict(definition=lab, event_rate=round(yy.mean(), 3),
                          AUC=round(roc_auc_score(yy[te4], pte), 3),
                          lo=round(boot_ci_auc(yy[te4], pte)[0], 3),
                          hi=round(boot_ci_auc(yy[te4], pte)[1], 3)))
sensA = pd.DataFrame(sens_rows)
print('=== 四敏感性表型AUC ==='); print(sensA.to_string(index=False))
print('(v4: 0.733 / 0.757 / 0.733 / 0.740)')
sensA.to_csv(f'{D}/results/v5_sensitivity_S1_definitions.csv', index=False)

# IPW加权OOF AUC（61流程：含缺失指示的PS模型）
conn = psycopg2.connect(dbname='mimic', host='127.0.0.1', port=5431,
                        user='postgres', password='<SET-IN-ENVIRONMENT>')
full2 = pd.read_sql("""SELECT stay_id, age, gender, sofa_24h, heart_rate_max, mbp_min,
                       resp_rate_max, spo2_min, lactate_max_24h, wbc_max, creatinine_max,
                       invasive_vent, cci, evaluable
                       FROM mimiciv_echo.sicm_analysis_dataset
                       WHERE age>=18 AND age<=90 AND has_acs=0""", conn)
conn.close()
Xps2 = full2[PSF].copy()
for c in PSF:
    Xps2[c + '_miss'] = Xps2[c].isna().astype(int)
    Xps2[c] = Xps2[c].fillna(Xps2[c].median())
Xps2['invasive_vent'] = full2['invasive_vent'].fillna(0)
Xps2['gender_f'] = (full2['gender'] == 'F').astype(int)
sc2 = StandardScaler().fit(Xps2)
ps2 = LogisticRegression(max_iter=2000).fit(sc2.transform(Xps2), full2['evaluable']) \
        .predict_proba(sc2.transform(Xps2))[:, 1]
prev2 = full2['evaluable'].mean()
w2 = np.where(full2['evaluable'] == 1, prev2 / ps2, (1 - prev2) / (1 - ps2))
lo2w, hi2w = np.percentile(w2[full2['evaluable'] == 1], [1, 99])
full2['sw'] = np.where(full2['evaluable'] == 1, np.clip(w2, lo2w, hi2w), 1.0)
legw = leg.merge(full2[['stay_id', 'sw']], on='stay_id', how='left')
wauc = roc_auc_score(yl, oofL, sample_weight=legw['sw'].fillna(1).values)
print(f'IPW加权OOF AUC: {oofL_auc:.3f} -> {wauc:.3f} (v4: 0.743->0.739)')
pd.DataFrame({'metric': ['oof_auc', 'weighted_oof_auc'],
              'value': [round(oofL_auc, 3), round(wauc, 3)]}).to_csv(
    f'{D}/results/v5_weighted_auc.csv', index=False)

# 最差校准亚组的交叉重校准（63流程）
def ici8(yy, pp):
    pc = calibration_curve(yy, pp, n_bins=8, strategy='quantile')
    return float(np.mean(np.abs(pc[1] - pc[0])))
cand = []
for dim, lst in subgroups.items():
    for lab, mask in lst:
        m = mask.values
        if m.sum() >= 60 and yl[m].sum() >= 20:
            cand.append((lab, ici8(yl[m], oofL[m])))
cand.sort(key=lambda x: -x[1])
worst_lab, worst_ici = cand[0]
wmask = None
for dim, lst in subgroups.items():
    for lab, mask in lst:
        if lab == worst_lab:
            wmask = mask.values
pc_l = np.clip(oofL[wmask], 1e-6, 1 - 1e-6)
lp = np.log(pc_l / (1 - pc_l))
yyw = yl[wmask]
idx = np.arange(len(yyw)); rng.shuffle(idx)
i1, i2 = idx[:len(idx) // 2], idx[len(idx) // 2:]
p_recal = np.zeros(len(yyw))
for a_, b_ in [(i1, i2), (i2, i1)]:
    f_ = sm.Logit(yyw[a_], sm.add_constant(lp[a_])).fit(disp=0)
    p_recal[b_] = f_.predict(sm.add_constant(lp[b_]))
ici_post = ici8(yyw, p_recal)
print(f'最差校准亚组: {worst_lab}, ICI {worst_ici:.3f} -> {ici_post:.3f} (v4: 0.098->0.057)')
pd.DataFrame({'item': ['worst_subgroup', 'ici_pre', 'ici_post'],
              'value': [worst_lab, round(worst_ici, 3), round(ici_post, 3)]}).to_csv(
    f'{D}/results/v5_recalibration.csv', index=False)

# ============ Part 8: 编码年代分层OR（严格CONF） ============
print('\n=== Part8 年代分层(严格CONF) ===')
for tag, dd in [('ICD-9期(2008-2013)', dn[dn.anchor_year_group.isin(['2008 - 2010', '2011 - 2013'])]),
                ('跨年代(2014-2016)', dn[dn.anchor_year_group.eq('2014 - 2016')]),
                ('ICD-10期(2017-2022)', dn[dn.anchor_year_group.isin(['2017 - 2019', '2020 - 2022'])])]:
    if len(dd) < 30:
        continue
    oo, llo, hhi, ppv = adj_or(dd, 'sicm')
    print(f'{tag}: n={len(dd)}, SICM={int(dd.sicm.sum())}, adj OR={oo:.2f} ({llo:.2f}-{hhi:.2f}), P={ppv:.3f}')
print('[done all]')

