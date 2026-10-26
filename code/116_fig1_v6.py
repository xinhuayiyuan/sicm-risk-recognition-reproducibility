# 116_fig1_v6.py — Figure 1: cohort derivation plus analysis-set register.
# Every analysis set used in the manuscript is labelled with n/events and its reporting location.
# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

D = '<PROJECT_ROOT>/exec'
fig = plt.figure(figsize=(10.5, 12.5))
gs = fig.add_gridspec(2, 1, height_ratios=[0.86, 1.14], hspace=0.12)
axA = fig.add_subplot(gs[0])
axB = fig.add_subplot(gs[1])
for ax in (axA, axB):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')


def box(ax, x, y, w, h, text, fc='#eaf2fb', ec='#1f77b4', fs=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.010',
                                fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs,
            linespacing=1.15)


def arrow(ax, xy1, xy2, color='#333333'):
    ax.add_patch(FancyArrowPatch(xy1, xy2, arrowstyle='-|>', mutation_scale=15,
                                 color=color, lw=1.3))


axA.text(0.01, 0.98, 'A. Cohort derivation', fontsize=15, fontweight='bold',
         ha='left', va='top')
box(axA, 0.10, 0.80, 0.44, 0.12, 'Sepsis-3 ICU stays (MIMIC-IV 2008–2022)\nn = 18,273', fs=10)
box(axA, 0.10, 0.60, 0.44, 0.12, 'Echocardiography + troponin evaluable\nn = 2,350', fs=10)
box(axA, 0.10, 0.40, 0.44, 0.12, 'Descriptive cohort (Table 1)\nn = 1,783', fs=10)
box(axA, 0.06, 0.10, 0.45, 0.19,
    'No HF history\nNew-onset cohort: n = 789\n215 SICM  |  574 reference', fs=10)
box(axA, 0.59, 0.10, 0.36, 0.19,
    'Pre-existing heart disease\nn = 994 (625 SICM)', fc='#f5f5f5', ec='#777777', fs=9.5)
box(axA, 0.60, 0.80, 0.35, 0.12, 'Excluded: no troponin\nor no LVEF echo\nn = 15,923',
    fc='#fdf0e8', ec='#d62728', fs=9)
box(axA, 0.60, 0.60, 0.35, 0.12, 'Excluded: age / ACS /\nprior cardiac surgery\nn = 567',
    fc='#fdf0e8', ec='#d62728', fs=9)
arrow(axA, (0.32, 0.80), (0.32, 0.72))
arrow(axA, (0.32, 0.60), (0.32, 0.52))
arrow(axA, (0.32, 0.40), (0.28, 0.29))
arrow(axA, (0.32, 0.40), (0.77, 0.29), color='#777777')
arrow(axA, (0.54, 0.86), (0.60, 0.86))
arrow(axA, (0.54, 0.66), (0.60, 0.66))
axA.text(0.01, 0.035, 'The 215 SICM cases are an outcome subgroup of the 789, not an independent cohort.',
         fontsize=9, color='#333333')
axA.text(0.01, 0.005, 'Additional, non-derivation analysis sets are listed in panel B.',
         fontsize=9, color='#444444', style='italic')

axB.text(0.01, 0.99, 'B. Analysis-set register: n, events and use in this manuscript',
         fontsize=15, fontweight='bold', ha='left', va='top')
axB.text(0.00, 0.93, 'Analysis set', fontsize=11.5, fontweight='bold', ha='left', va='center')
axB.text(0.245, 0.93, 'n (events)', fontsize=11.5, fontweight='bold', ha='left', va='center')
axB.text(0.385, 0.93, 'Role in this manuscript (tables and figures)',
         fontsize=11.5, fontweight='bold', ha='left', va='center')
axB.plot([0, 1], [0.905, 0.905], color='black', lw=1.1)

rows = [
    ('Descriptive cohort', '1,783\n(840 SICM)', 'Baseline description (Table 1)'),
    ('New-onset cohort', '789\n(215 SICM; 574 ref)', 'Primary model, mortality, mediation and SHAP\n(Tables 2–4; Figs 2–4)'),
    ('Development split', '552\n(150 events)', 'LASSO fitting only (Fig. 2)'),
    ('Held-out test set', '237\n(65 events)', 'Discrimination, calibration, DCA and exploratory threshold\n(Table 2; Fig. 2; Supplementary Fig. S4A)'),
    ('Five-fold OOF', '789\n(215 events)', 'Internal validation (Table 2)'),
    ('24-h landmark subset', '610\n(60 events)', 'Post-24 h transport sensitivity (Supplementary Fig. S4B)'),
    ('ICD-10-only HF cohort', '1,504\n(676 SICM)', 'HF-coding sensitivity and subgroups\n(Supplementary Table S1; Supplementary Fig. S3)'),
    ('Echo-selection source', '16,462\n(1,783 evaluable)', 'IPW fit and balance; ESS 1,254 of 1,783\n(Table 3 note; Supplementary Fig. S2)'),
    ('eICU proxy-pathway cohort', '6,006\n(3,902 exposed)', 'Exploratory pathway only; not external validation\n(Table 4; Fig. 4A; Supplementary Fig. S1)'),
]
y0, step = 0.845, 0.092
for k, (name, n, use) in enumerate(rows):
    y = y0 - k * step
    if k % 2 == 0:
        axB.add_patch(plt.Rectangle((0, y - 0.041), 1.0, 0.082, color='#eef3f8', lw=0))
    axB.text(0.00, y, name, fontsize=10.5, ha='left', va='center')
    axB.text(0.245, y, n, fontsize=10.0, ha='left', va='center', linespacing=1.05)
    axB.text(0.385, y, use, fontsize=10.0, ha='left', va='center', linespacing=1.05)
axB.plot([0, 1], [0.055, 0.055], color='black', lw=1.1)
axB.text(0.00, 0.012,
         'SICM, sepsis-induced cardiomyopathy; HF, heart failure; OOF, out-of-fold; DCA, decision-curve analysis;\n'
         'IPW, inverse probability weighting; ESS, effective sample size.',
         fontsize=9.2, color='#333333', va='bottom')

fig.subplots_adjust(left=0.02, right=0.985, top=0.985, bottom=0.015, hspace=0.12)
fig.savefig(f'{D}/figures/fig1_cohort_flow.png', dpi=220, bbox_inches='tight')
plt.close(fig)
print('[done 116: fig1 cohort derivation + analysis-set register saved]')
