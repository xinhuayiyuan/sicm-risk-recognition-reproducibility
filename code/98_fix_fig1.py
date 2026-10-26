# 98_fix_fig1.py - 图1重绘：大字号 + 右侧排除框完整可见
# -*- coding: utf-8 -*-
import textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

D = '<PROJECT_ROOT>/exec'
fig, ax = plt.subplots(figsize=(11.5, 9.5))
ax.axis('off')

def wrap(t, w):
    return '\n'.join(textwrap.fill(x, w) for x in t.split('\n'))

boxes = [
    ("Sepsis-3 ICU stays\n(MIMIC-IV 2008-2022)\nn = 18,273", 0.92),
    ("Evaluable\n(troponin + LVEF echo)\nn = 2,350", 0.74),
    ("Primary cohort\n(age 18-90, no ACS)\nn = 1,783", 0.56),
    ("New-onset modelling cohort\n(no HF by ICD-9 428.x AND ICD-10 I50)\nn = 789", 0.38),
    ("New-onset LV systolic SICM\nn = 215 (27.2%)", 0.20),
]
excl = [
    ("Excluded:\nno troponin or no echo\n(n = 15,923)", 0.83),
    ("Excluded: age / ACS /\nprior cardiac surgery (n = 567)", 0.65),
    ("HF history by either coding system\n(n = 994, incl. 715 ICD-9-era HF)", 0.47),
]
# 主框：居中，加大加宽
for txt, ypos in boxes:
    ax.add_patch(FancyBboxPatch((0.18, ypos - 0.062), 0.42, 0.115,
                                boxstyle="round,pad=0.012", fc='#eaf2fb', ec='#1f77b4', lw=1.4))
    ax.text(0.39, ypos, wrap(txt, 24), ha='center', va='center', fontsize=12)
# 右侧排除框：确保完全在画布内（x+宽 <= 1）
for txt, ypos in excl:
    ax.add_patch(FancyBboxPatch((0.66, ypos - 0.062), 0.32, 0.115,
                                boxstyle="round,pad=0.012", fc='#fdf0e8', ec='#d62728', lw=1.2))
    ax.text(0.82, ypos, wrap(txt, 24), ha='center', va='center', fontsize=10)
ys = [b[1] for b in boxes]
for i in range(len(ys) - 1):
    ax.add_patch(FancyArrowPatch((0.39, ys[i] - 0.062), (0.39, ys[i + 1] + 0.062),
                                 arrowstyle='-|>', mutation_scale=20, color='#333333', lw=1.5))
for _, ypos in excl:
    ax.add_patch(FancyArrowPatch((0.60, ypos), (0.66, ypos),
                                 arrowstyle='-|>', mutation_scale=15, color='#d62728', lw=1.2))
ax.set_xlim(0, 1)
ax.set_ylim(0.11, 1.0)
fig.tight_layout()
fig.savefig(f'{D}/figures/fig1_cohort_flow.png', dpi=200)
plt.close('all')
print('fig1 redrawn:', f'{D}/figures/fig1_cohort_flow.png')
