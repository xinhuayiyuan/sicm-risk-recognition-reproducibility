# 90_additional_figures.py - 增补图件（全部基于已计算结果）
# Fig2: ROC+校准曲线面板（修正队列测试集） | Fig6: 中介效应森林图 | FigS: 亚组AUC森林图
# -*- coding: utf-8 -*-
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.calibration import calibration_curve

SEED = 20260101
D = '<PROJECT_ROOT>/exec'
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
means = las.scores_[1].mean(axis=0); ses = las.scores_[1].std(axis=0) / np.sqrt(10)
c1se = las.Cs_[np.where(means >= means.max() - ses[means.argmax()])[0][0]]
las1 = LogisticRegression(C=c1se, penalty='l1', solver='liblinear',
                          max_iter=3000, random_state=SEED).fit(Xtr, ytr)
mb = LogisticRegression(max_iter=1000).fit(dtr[['age', 'sofa_24h']], ytr)
preds = [('LASSO (lambda.min)', las.predict_proba(Xte)[:, 1], '#d62728'),
         ('LASSO (1-SE, 21 vars)', las1.predict_proba(Xte)[:, 1], '#ff7f0e'),
         ('Baseline (age+SOFA)', mb.predict_proba(dte[['age', 'sofa_24h']])[:, 1], '#7f7f7f')]

# ---- Fig2: ROC + 校准 面板 ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))
for name, p, c in preds:
    fpr, tpr, _ = roc_curve(yte, p)
    ax1.plot(fpr, tpr, color=c, label=f"{name} (AUC {roc_auc_score(yte, p):.3f})")
ax1.plot([0, 1], [0, 1], 'k--', lw=0.8)
ax1.set_xlabel('False positive rate'); ax1.set_ylabel('True positive rate')
ax1.set_title('A. ROC curves (test set)'); ax1.legend(loc='lower right', fontsize=8.5); ax1.grid(alpha=0.3)
ax2.plot([0, 1], [0, 1], 'k--', lw=0.8, label='Perfect calibration')
for name, p, c in preds:
    pc = calibration_curve(yte, p, n_bins=10, strategy='quantile')
    ax2.plot(pc[0], pc[1], 'o-', color=c, label=name, ms=4)
ax2.set_xlabel('Mean predicted probability'); ax2.set_ylabel('Observed event rate')
ax2.set_title('B. Calibration (10 quantile bins)'); ax2.legend(loc='upper left', fontsize=8.5); ax2.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f'{D}/figures/fig2_roc_calibration_corrected.png', dpi=200)
print('fig2 saved')


# ---- Fig6: 中介效应森林图 ----
meds = []
for f, lab in [(f'{D}/results/v4_corrected_mediation_vaso_new_upgrade.csv', 'MIMIC: vasopressor initiation/up-titration'),
               (f'{D}/results/v4_corrected_mediation_log_ned.csv', 'MIMIC: peak norepinephrine (log)'),
               (f'{D}/results/aim_eicu_V3_mediation.csv', 'eICU: norepinephrine use (proxy exposure)')]:
    m = pd.read_csv(f).set_index('effect')
    meds.append((lab, m))
fig, ax = plt.subplots(figsize=(8, 5.5))
yticks = []
ylabels = []
ypos = 0
colors = {'TE': '#1f77b4', 'NDE': '#2ca02c', 'NIE': '#d62728'}
for lab, m in meds:
    ylabels.append((ypos + 1.5, lab))
    for eff in ['TE', 'NDE', 'NIE']:
        est, lo, hi = m.loc[eff, 'estimate'], m.loc[eff, 'ci_lo'], m.loc[eff, 'ci_hi']
        ax.errorbar(est, ypos, xerr=[[est - lo], [hi - est]], fmt='o', color=colors[eff],
                    capsize=3, ms=6)
        ax.text(hi + 0.004, ypos, f"{est*100:+.1f} ({lo*100:+.1f} to {hi*100:+.1f})",
                va='center', fontsize=8)
        yticks.append((ypos, eff))
        ypos += 1
    ypos += 1.5
ax.axvline(0, color='k', ls='--', lw=0.8)
ax.set_yticks([t[0] for t in yticks]); ax.set_yticklabels([t[1] for t in yticks])
for yv, lab in ylabels:
    ax.text(ax.get_xlim()[0], yv, lab, fontweight='bold', fontsize=9, va='bottom')
ax.set_xlabel('Effect on 28-day mortality (risk difference)')
ax.set_xlim(-0.08, 0.24)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([0], [0], marker='o', color=colors[k], label=l, ls='')
                   for k, l in [('TE', 'Total effect'), ('NDE', 'Direct effect'), ('NIE', 'Indirect effect')]],
          loc='lower right', fontsize=8.5)
ax.set_title('Mediation of new-onset SICM effect on 28-day mortality')
ax.grid(alpha=0.3, axis='x')
fig.tight_layout(); fig.savefig(f'{D}/figures/fig6_mediation_forest.png', dpi=200)
print('fig6 saved')

# ---- FigS: 亚组AUC森林图（HF未限制队列，补充材料用） ----
sg = pd.read_csv(f'{D}/results/aim2_subgroup_heterogeneity.csv')
sg = sg.dropna(subset=['AUC'])
sg['lab'] = sg['维度'].astype(str) + ': ' + sg['亚组'].astype(str)
sg = sg.iloc[::-1]
fig, ax = plt.subplots(figsize=(7.5, 5.5))
ax.errorbar(sg['AUC'], range(len(sg)),
            xerr=[sg['AUC'] - sg['AUC_lo'], sg['AUC_hi'] - sg['AUC']],
            fmt='o', color='#1f77b4', capsize=2.5, ms=5)
ax.axvline(0.743, color='#d62728', ls='--', lw=1, label='Overall OOF AUC (0.743)')
ax.set_yticks(range(len(sg))); ax.set_yticklabels(sg['lab'], fontsize=8.5)
ax.set_xlabel('AUC (95% CI)'); ax.set_xlim(0.5, 0.95)
ax.legend(loc='lower right', fontsize=8.5); ax.grid(alpha=0.3, axis='x')
ax.set_title('Subgroup discrimination (HF-unrestricted cohort, out-of-fold)')
fig.tight_layout(); fig.savefig(f'{D}/figures/figS_subgroup_forest.png', dpi=200)
print('figS saved')
print('[done]')
