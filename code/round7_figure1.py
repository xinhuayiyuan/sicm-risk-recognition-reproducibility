# 116_fig1_v6.py — Figure 1: MIMIC cohort derivation and primary analysis population.
# The complete analysis-set register is an editable table in Supplementary Table S1.
# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

D = '<PROJECT_ROOT>/exec'
fig, ax = plt.subplots(figsize=(10.5, 7.8))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')


def box(x, y, w, h, text, fc='#eaf2fb', ec='#1f77b4', fs=11):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.010',
                                fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs,
            linespacing=1.2)


def arrow(xy1, xy2, color='#333333'):
    ax.add_patch(FancyArrowPatch(xy1, xy2, arrowstyle='-|>', mutation_scale=16,
                                 color=color, lw=1.35))


ax.text(0.02, 0.98, 'MIMIC-IV cohort derivation', fontsize=16, fontweight='bold',
        ha='left', va='top')
box(0.08, 0.79, 0.48, 0.12,
    'Sepsis-3 ICU stays (MIMIC-IV, 2008–2022)\nn = 18,273')
box(0.08, 0.57, 0.48, 0.12,
    'Troponin measurement and LVEF-documented\nechocardiography available: n = 2,350')
box(0.08, 0.35, 0.48, 0.12,
    'Descriptive cohort (Table 1)\nn = 1,297')
box(0.04, 0.08, 0.47, 0.16,
    'No index-admission HF diagnosis\nNew-onset cohort: n = 613\n111 new-onset SICM  |  502 reference')
box(0.60, 0.08, 0.36, 0.16,
    'Index-admission HF diagnosis\nn = 684 (355 with SICM)',
    fc='#f5f5f5', ec='#777777', fs=10.5)
box(0.64, 0.79, 0.32, 0.12,
    'Excluded: no troponin or no\nLVEF echocardiography\nn = 15,923',
    fc='#fdf0e8', ec='#d62728', fs=10)
box(0.64, 0.57, 0.32, 0.12,
    'Excluded: age / ACS\nn = 1,053',
    fc='#fdf0e8', ec='#d62728', fs=9.5)
arrow((0.32, 0.79), (0.32, 0.69))
arrow((0.32, 0.57), (0.32, 0.47))
arrow((0.32, 0.35), (0.275, 0.24))
arrow((0.32, 0.35), (0.78, 0.24), color='#777777')
arrow((0.56, 0.85), (0.64, 0.85))
arrow((0.56, 0.63), (0.64, 0.63))
ax.text(0.02, 0.025,
        'ACS, acute coronary syndrome; HF, heart failure; LVEF, left ventricular ejection fraction; '
        'SICM, sepsis-induced cardiomyopathy.',
        fontsize=9.5, color='#333333', ha='left', va='bottom')
ax.text(0.02, 0.002,
        'The 111 SICM cases are an outcome-defined subgroup of the 613-stay new-onset cohort.',
        fontsize=9.5, color='#333333', ha='left', va='bottom', style='italic')

fig.subplots_adjust(left=0.02, right=0.985, top=0.985, bottom=0.04)
fig.savefig(f'{D}/figures/fig1_cohort_flow.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print('[done 116: single-panel MIMIC cohort flow saved; register moved to Supplementary Table S1]')
