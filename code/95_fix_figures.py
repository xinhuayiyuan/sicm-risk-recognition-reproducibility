# 95_fix_figures.py - 图件硬伤修复
# Fig4 SHAP净版(独立图窗) | FigS1英文标签 | Fig3A森林图(布局+舍入) | Fig3B实测剂量分组图(替代样条)
# -*- coding: utf-8 -*-
import warnings; warnings.filterwarnings('ignore')
from decimal import Decimal, ROUND_HALF_UP
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler
from scipy import stats
import xgboost as xgb
import shap

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
X = dn[FEATS].fillna(dn[FEATS].median())
y = dn['sicm'].values

# ============ ① Fig4 SHAP 净版（独立图窗，杜绝残层） ============
sc_ = StandardScaler().fit(X[NUM])
Xs = X.copy(); Xs[NUM] = sc_.transform(Xs[NUM])
mx = xgb.train(dict(objective='binary:logistic', eval_metric='auc', max_depth=3, eta=0.05,
                    subsample=0.8, colsample_bytree=0.8, min_child_weight=20, seed=SEED),
               xgb.DMatrix(Xs, label=y), num_boost_round=400, verbose_eval=False)
sv = shap.TreeExplainer(mx).shap_values(Xs)
plt.figure(figsize=(8, 6))                      # 关键修复：独立新图窗
shap.summary_plot(sv, Xs, feature_names=FEATS, show=False, max_display=15)
plt.tight_layout()
plt.savefig(f'{D}/figures/fig4_shap_corrected.png', dpi=200, bbox_inches='tight')
plt.close('all')
print('fig4 clean saved')

# ============ ② FigS1 亚组森林图（英文标签） ============
sg = pd.read_csv(f'{D}/results/aim2_subgroup_heterogeneity.csv').dropna(subset=['AUC'])
EN = {'年龄: 年龄<65': 'Age <65 years', '年龄: 65-74': 'Age 65–74 years',
      '年龄: >=75': 'Age ≥75 years', 'SOFA: SOFA<8': 'SOFA <8', 'SOFA: SOFA>=8': 'SOFA ≥8',
      '休克(24h NE): 未用NE': 'No norepinephrine in 24 h',
      '休克(24h NE): 已用NE': 'Norepinephrine in 24 h',
      'AKI: AKI 0': 'AKI stage 0', 'AKI: AKI 1': 'AKI stage 1',
      'AKI: AKI 2': 'AKI stage 2', 'AKI: AKI 3': 'AKI stage 3',
      '机械通气: 未通气': 'No invasive ventilation', '机械通气: 有创通气': 'Invasive ventilation'}
sg['lab'] = (sg['维度'].astype(str) + ': ' + sg['亚组'].astype(str)).map(EN)
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
plt.close('all')
print('figS1 EN saved')


# ============ ③④ Fig3 双联图（A: 中介森林[舍入统一+不遮挡] B: 实测剂量分组[替代样条]） ============
def fmt(v):
    dv = Decimal(str(v * 100))
    if abs(dv) < 1:
        return f"{dv.quantize(Decimal('0.01'), ROUND_HALF_UP):+}"
    return f"{dv.quantize(Decimal('0.1'), ROUND_HALF_UP):+}"

meds = []
for f, lab in [(f'{D}/results/v4_corrected_mediation_vaso_new_upgrade.csv', 'MIMIC: vasopressor initiation'),
               (f'{D}/results/v4_corrected_mediation_log_ned.csv', 'MIMIC: peak norepinephrine (log)'),
               (f'{D}/results/aim_eicu_V3_mediation.csv', 'eICU: norepinephrine use (proxy)')]:
    meds.append((lab, pd.read_csv(f).set_index('effect')))
colors = {'TE': '#1f77b4', 'NDE': '#2ca02c', 'NIE': '#d62728'}

def dose_groups(dsub):
    ned = dsub['post_max_ned'].fillna(0).values.astype(float)
    yy = dsub['mort_28d'].values.astype(float)
    out = [('No norepinephrine', yy[ned == 0])]
    t = pd.qcut(pd.Series(ned[ned > 0]), 3, duplicates='drop')
    for i, cat in enumerate(t.cat.categories):
        out.append((f'T{i+1}', yy[ned > 0][(t == cat).values]))
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
plt.close('all')
print('fig3 composite saved')
print('[done]')
