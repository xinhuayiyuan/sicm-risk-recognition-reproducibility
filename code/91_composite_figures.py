# 91_composite_figures.py - 图1溢框修复重绘 + 复合图合并
# Fig2 = 模型性能三联图(ROC/校准/DCA) | Fig3 = 路径双联图(中介森林/剂量-反应)
# -*- coding: utf-8 -*-
import warnings; warnings.filterwarnings('ignore')
import textwrap
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.calibration import calibration_curve
from patsy import dmatrix
import statsmodels.api as sm

SEED = 20260101
D = '<PROJECT_ROOT>/exec'

# ============ ① Fig1 重绘（修复文字溢框） ============
fig, ax = plt.subplots(figsize=(9, 9.5)); ax.axis('off')
def wrap(t, w):
    return '\n'.join(textwrap.fill(x, w) for x in t.split('\n'))
boxes = [
    ("Sepsis-3 ICU stays\n(MIMIC-IV 2008-2022)\nn = 18,273", 0.93),
    ("Evaluable\n(troponin + LVEF echo)\nn = 2,350", 0.755),
    ("Primary cohort\n(age 18-90, no ACS)\nn = 1,783", 0.58),
    ("New-onset modelling cohort\n(no HF by ICD-9 428.x AND ICD-10 I50)\nn = 789", 0.405),
    ("New-onset LV systolic SICM\nn = 215 (27.2%)", 0.23),
]
excl = [
    ("Excluded: no troponin or no echo (n = 15,923)", 0.845),
    ("Excluded: age / ACS / prior cardiac surgery (n = 567)", 0.67),
    ("HF history by either coding system (n = 994, incl. 715 ICD-9-era CHF)", 0.495),
]
for txt, ypos in boxes:
    ax.add_patch(FancyBboxPatch((0.24, ypos - 0.06), 0.46, 0.115,
                                boxstyle="round,pad=0.012", fc='#eaf2fb', ec='#1f77b4'))
    ax.text(0.47, ypos, wrap(txt, 30), ha='center', va='center', fontsize=9.5)
for txt, ypos in excl:
    ax.add_patch(FancyBboxPatch((0.75, ypos - 0.055), 0.24, 0.105,
                                boxstyle="round,pad=0.012", fc='#fdf0e8', ec='#d62728'))
    ax.text(0.87, ypos, wrap(txt, 26), ha='center', va='center', fontsize=7.5)
ys = [b[1] for b in boxes]
for i in range(len(ys) - 1):
    ax.add_patch(FancyArrowPatch((0.47, ys[i] - 0.06), (0.47, ys[i + 1] + 0.06),
                                 arrowstyle='-|>', mutation_scale=18, color='#333333'))
for _, ypos in excl:
    ax.add_patch(FancyArrowPatch((0.70, ypos), (0.75, ypos),
                                 arrowstyle='-|>', mutation_scale=13, color='#d62728'))
ax.set_xlim(0, 1); ax.set_ylim(0.14, 1.0)
fig.tight_layout(); fig.savefig(f'{D}/figures/fig1_cohort_flow.png', dpi=200)
print('fig1 fixed')

# ============ 数据准备（修正队列，与脚本86/90同流程） ============
co = pd.read_csv(f'{D}/data/cohort_main.csv')
st = pd.read_csv(f'{D}/data/cohort_dn_strict24h.csv',
                 usecols=['stay_id', 'ne_max_24h_strict', 'dobutamine_24h',
                          'milrinone_24h', 'vent_24h'])
dn = co[(co['has_hf'] == 0) & (co['congestive_heart_failure'] == 0)].merge(
    st, on='stay_id', how='left').reset_index(drop=True)
NUM = ['age','cci','sofa_24h','heart_rate_max','mbp_min','resp_rate_max','spo2_min',
       'temperature_max','lactate_max_24h','wbc_max','platelets_min','creatinine_max',
       'bun_max','bilirubin_total_max','sodium_min','potassium_max','glucose_max',
       'inr_max','ptt_max','bicarbonate_min','aniongap_max','hemoglobin_min',
       'ne_max_24h_strict','aki_stage_max']
COMORB = ['myocardial_infarct','peripheral_vascular_disease','cerebrovascular_disease',
          'dementia','chronic_pulmonary_disease','rheumatic_disease','mild_liver_disease',
          'diabetes_without_cc','diabetes_with_cc','renal_disease','malignant_cancer',
          'severe_liver_disease','metastatic_solid_tumor','aids']
BIN = ['gender_f','race_black','race_other'] + COMORB + \
      ['vent_24h','dobutamine_24h','milrinone_24h','rrt_firstday','lactate_miss','bili_miss']
dn['gender_f'] = (dn['gender'] == 'F').astype(int)
rc = dn['race'].astype(str).str.upper()
dn['race_black'] = rc.str.contains('BLACK').astype(int)
dn['race_other'] = (~rc.str.contains('WHITE|BLACK')).astype(int)
for c in COMORB + ['rrt_firstday']:
    dn[c] = dn[c].fillna(0).astype(int)
FEATS = [f for f in (NUM + BIN) if f != 'aids']
y = dn['sicm'].values
tr, te = train_test_split(dn.index, test_size=0.3, stratify=y, random_state=SEED)
dtr, dte = dn.loc[tr].copy(), dn.loc[te].copy()
med = dtr[NUM].median()
dtr[NUM] = dtr[NUM].fillna(med); dte[NUM] = dte[NUM].fillna(med)
ytr, yte = dtr['sicm'].values, dte['sicm'].values
sc = StandardScaler().fit(dtr[NUM])
Xtr = dtr[FEATS].copy(); Xte = dte[FEATS].copy()
Xtr[NUM] = sc.transform(Xtr[NUM]); Xte[NUM] = sc.transform(Xte[NUM])
las = LogisticRegressionCV(Cs=20, penalty='l1', solver='liblinear', cv=10,
                           scoring='roc_auc', max_iter=3000, random_state=SEED).fit(Xtr, ytr)
mb = LogisticRegression(max_iter=1000).fit(dtr[['age', 'sofa_24h']], ytr)
preds = [('LASSO (lambda.min)', las.predict_proba(Xte)[:, 1], '#d62728'),
         ('Baseline (age+SOFA)', mb.predict_proba(dte[['age', 'sofa_24h']])[:, 1], '#7f7f7f')]

# ============ ② Fig2 模型性能三联图 (A:ROC B:校准 C:DCA) ============
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.5, 4.6))
for name, p, c in preds:
    fpr, tpr, _ = roc_curve(yte, p)
    ax1.plot(fpr, tpr, color=c, label=f"{name} (AUC {roc_auc_score(yte, p):.3f})")
ax1.plot([0, 1], [0, 1], 'k--', lw=0.8)
ax1.set_xlabel('False positive rate'); ax1.set_ylabel('True positive rate')
ax1.set_title('A. ROC curves'); ax1.legend(loc='lower right', fontsize=8); ax1.grid(alpha=0.3)
ax2.plot([0, 1], [0, 1], 'k--', lw=0.8, label='Perfect calibration')
for name, p, c in preds:
    pc = calibration_curve(yte, p, n_bins=10, strategy='quantile')
    ax2.plot(pc[0], pc[1], 'o-', color=c, label=name.split(' (')[0], ms=4)
ax2.set_xlabel('Mean predicted probability'); ax2.set_ylabel('Observed event rate')
ax2.set_title('B. Calibration (10 quantile bins)'); ax2.legend(loc='upper left', fontsize=8); ax2.grid(alpha=0.3)
dca = pd.read_csv(f'{D}/results/v4_corrected_dca.csv')
for name, c, ls in [('LASSO lambda.min', '#d62728', '-'), ('Baseline', '#7f7f7f', '-'),
                    ('Treat all', '#1f77b4', '--'), ('Treat none', '#000000', '--')]:
    dd = dca[dca.model == name]
    ax3.plot(dd.threshold, dd.net_benefit, label=name, color=c, ls=ls)
ax3.set_xlabel('Threshold probability'); ax3.set_ylabel('Net benefit')
ax3.set_ylim(-0.05, 0.3); ax3.set_title('C. Decision-curve analysis')
ax3.legend(fontsize=8); ax3.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f'{D}/figures/fig2_model_performance.png', dpi=200)
print('fig2 composite saved')



# ============ ③ Fig3 路径双联图 (A:中介森林 B:剂量-反应) ============
fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.2),
                               gridspec_kw={'width_ratios': [1.35, 1]})
meds = []
for f, lab in [(f'{D}/results/v4_corrected_mediation_vaso_new_upgrade.csv', 'MIMIC: vasopressor initiation'),
               (f'{D}/results/v4_corrected_mediation_log_ned.csv', 'MIMIC: peak norepinephrine (log)'),
               (f'{D}/results/aim_eicu_V3_mediation.csv', 'eICU: norepinephrine use (proxy)')]:
    meds.append((lab, pd.read_csv(f).set_index('effect')))
yticks, ylabels, ypos = [], [], 0
colors = {'TE': '#1f77b4', 'NDE': '#2ca02c', 'NIE': '#d62728'}
for lab, m in meds:
    ylabels.append((ypos + 1.5, lab))
    for eff in ['TE', 'NDE', 'NIE']:
        est, lo, hi = m.loc[eff, 'estimate'], m.loc[eff, 'ci_lo'], m.loc[eff, 'ci_hi']
        axA.errorbar(est, ypos, xerr=[[est - lo], [hi - est]], fmt='o',
                     color=colors[eff], capsize=3, ms=5.5)
        axA.text(hi + 0.004, ypos, f"{est*100:+.1f} ({lo*100:+.1f} to {hi*100:+.1f})",
                 va='center', fontsize=7.5)
        yticks.append((ypos, eff)); ypos += 1
    ypos += 1.5
axA.axvline(0, color='k', ls='--', lw=0.8)
axA.set_yticks([t[0] for t in yticks]); axA.set_yticklabels([t[1] for t in yticks])
for yv, lab in ylabels:
    axA.text(-0.075, yv, lab, fontweight='bold', fontsize=8.5, va='bottom')
axA.set_xlabel('Effect on 28-day mortality (risk difference)'); axA.set_xlim(-0.08, 0.26)
axA.legend(handles=[Line2D([0], [0], marker='o', color=colors[k], label=l, ls='')
                    for k, l in [('TE', 'Total effect'), ('NDE', 'Direct effect'),
                                 ('NIE', 'Indirect effect')]], loc='lower right', fontsize=8)
axA.set_title('A. Mediation effects (95% CI)'); axA.grid(alpha=0.3, axis='x')
for lab, mask, c in [('New-onset SICM', dn['sicm'] == 1, '#d62728'),
                     ('Reference', dn['sicm'] == 0, '#1f77b4')]:
    dsub = dn[mask]
    ned = dsub['post_max_ned'].fillna(0).values.astype(float)
    yy = dsub['mort_28d'].values.astype(float)
    Xs = dmatrix("cr(ned, df=4)", {'ned': ned}, return_type='dataframe')
    m = sm.Logit(yy, Xs).fit(disp=0)
    grid = np.linspace(0, np.percentile(ned, 99), 100)
    Xg = dmatrix("cr(ned, df=4)", {'ned': grid}, return_type='dataframe')
    axB.plot(grid, m.predict(Xg), color=c, label=f"{lab} (n={len(dsub)})")
axB.set_xlabel('Peak norepinephrine-equivalent dose')
axB.set_ylabel('Predicted 28-day mortality')
axB.set_title('B. Dose-response (unadjusted)'); axB.legend(fontsize=8); axB.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f'{D}/figures/fig3_pathway.png', dpi=200)
print('fig3 composite saved')
print('[done]')
