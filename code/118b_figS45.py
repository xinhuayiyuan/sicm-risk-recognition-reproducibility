# 118b_figS45.py — figS4(IPW Love plot) + figS5(eICU流程图)
# -*- coding: utf-8 -*-
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import textwrap
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

D = '<PROJECT_ROOT>/exec'

# ---- figS4: Love plot ----
smd = pd.read_csv(f'{D}/results/v6_ipw_balance_smd.csv')
smd = smd.reindex(smd['SMD_unweighted'].abs().sort_values().index)
fig, ax = plt.subplots(figsize=(7.5, 5.5))
yy = np.arange(len(smd))
ax.scatter(smd['SMD_unweighted'], yy, color='#d62728', label='Before weighting', zorder=3)
ax.scatter(smd['SMD_weighted'], yy, color='#1f77b4', label='After weighting', zorder=3)
for i, (u, w_) in enumerate(zip(smd['SMD_unweighted'], smd['SMD_weighted'])):
    ax.plot([u, w_], [i, i], color='grey', lw=0.8, zorder=1)
ax.axvline(0.1, color='k', ls='--', lw=0.8); ax.axvline(-0.1, color='k', ls='--', lw=0.8)
COV = {'age':'Age','sofa_24h':'SOFA score, first 24 h','heart_rate_max':'Heart rate max',
       'mbp_min':'MAP min','resp_rate_max':'Respiratory rate max','spo2_min':'SpO2 min',
       'lactate_max_24h':'Lactate max','wbc_max':'WBC max','creatinine_max':'Creatinine max',
       'cci':'Charlson comorbidity index','invasive_vent':'Invasive ventilation (whole stay)',
       'gender_f':'Female sex'}
ax.set_yticks(yy); ax.set_yticklabels([COV.get(x, x) for x in smd['covariate']], fontsize=9)
ax.set_xlabel('Standardised mean difference (evaluable vs non-evaluable)')
ax.set_title('Echocardiography-selection IPW source population: n = 16,462\nEvaluable: n = 1,783; non-evaluable: n = 14,679', fontsize=11)
ax.legend(fontsize=9, loc='lower right'); ax.grid(alpha=0.3, axis='x')
fig.tight_layout(); fig.savefig(f'{D}/figures/figS4_ipw_loveplot.png', dpi=200)
plt.close('all'); print('[figS4 saved]')

# ---- figS5: eICU流程 ----
fig, ax = plt.subplots(figsize=(9, 8))
ax.axis('off')
flow = [
    ("eICU-CRD v2.0 Sepsis-3 ICU stays\n(208 hospitals; age 18-90, ICU stay >=24 h)\nn = 18,253", 0.88),
    ("At least one troponin I measurement\n(eICU analysis cohort)\nn = 6,006", 0.62),
    ("Proxy exposure: no HF history +\ntroponin I >0.04 ng/mL (first 72 h)\nn = 3,902 (65.0%)", 0.36),
    ("Mediator: norepinephrine use (n = 462)\nOutcome: 28-day mortality (21.3%)\nTE +1.9 pp (0.1 to 4.1); no positive mediation", 0.12),
]
for txt, ypos in flow:
    ax.add_patch(FancyBboxPatch((0.16, ypos - 0.085), 0.68, 0.16,
                                boxstyle="round,pad=0.012", fc='#eaf2fb', ec='#1f77b4', lw=1.3))
    ax.text(0.5, ypos, '\n'.join(textwrap.fill(x, 46) for x in txt.split('\n')),
            ha='center', va='center', fontsize=11)
ys = [f[1] for f in flow]
for i in range(len(ys) - 1):
    ax.add_patch(FancyArrowPatch((0.5, ys[i] - 0.085), (0.5, ys[i + 1] + 0.085),
                                 arrowstyle='-|>', mutation_scale=18, color='#333333', lw=1.4))
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
fig.tight_layout(); fig.savefig(f'{D}/figures/figS5_eicu_flow.png', dpi=200)
plt.close('all'); print('[figS5 saved]')
print('[done 118b]')
