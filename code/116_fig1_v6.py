# 116_fig1_v6.py — C2: 图1重绘，补16,462(IPW分母)与1,504(ICD-10-only确认队列)两个侧支节点
# -*- coding: utf-8 -*-
import textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

D = '<PROJECT_ROOT>/exec'
fig, ax = plt.subplots(figsize=(13.5, 9.5))
ax.axis('off')

def wrap(t, w):
    return '\n'.join(textwrap.fill(x, w) for x in t.split('\n'))

# 主链（中列）
boxes = [
    ("Sepsis-3 ICU stays\n(MIMIC-IV 2008-2022)\nn = 18,273", 0.92),
    ("Evaluable\n(troponin + LVEF echo)\nn = 2,350", 0.74),
    ("Descriptive cohort: Table 1\n(age 18-90, no ACS)\nn = 1,783", 0.56),
    ("New-onset cohort: model, mortality and MIMIC mediation\n(no HF by ICD-9 428.x AND ICD-10 I50)\nn = 789", 0.38),
    ("New-onset LV systolic SICM\nn = 215 (27.2%)", 0.20),
]
# 右侧排除框
excl = [
    ("Excluded:\nno troponin or no echo\n(n = 15,923)", 0.83),
    ("Excluded: age / ACS /\nprior cardiac surgery (n = 567)", 0.65),
    ("HF history by either coding system\n(n = 994, incl. 715 ICD-9-era HF)", 0.47),
]
# 左侧分析队列支框
side = [
    ("Age/ACS-eligible stays\n(denominator of the echo-selection\nIPW analysis)\nn = 16,462", 0.83),
    ("ICD-10-only HF-ascertainment\ncohort (HF by ICD-10 I50 only\nexcluded, n = 279)\nn = 1,504", 0.47),
    ("Reference group\n(no HF history; no SICM)\nn = 574", 0.20),
]
for txt, ypos in boxes:
    ax.add_patch(FancyBboxPatch((0.30, ypos - 0.062), 0.40, 0.115,
                                boxstyle="round,pad=0.012", fc='#eaf2fb', ec='#1f77b4', lw=1.4))
    ax.text(0.50, ypos, wrap(txt, 30), ha='center', va='center', fontsize=12)
for txt, ypos in excl:
    ax.add_patch(FancyBboxPatch((0.74, ypos - 0.062), 0.25, 0.115,
                                boxstyle="round,pad=0.012", fc='#fdf0e8', ec='#d62728', lw=1.2))
    ax.text(0.865, ypos, wrap(txt, 22), ha='center', va='center', fontsize=9.5)
for txt, ypos in side:
    ax.add_patch(FancyBboxPatch((0.01, ypos - 0.062), 0.25, 0.115,
                                boxstyle="round,pad=0.012", fc='#eef7ee', ec='#2ca02c', lw=1.2))
    ax.text(0.135, ypos, wrap(txt, 24), ha='center', va='center', fontsize=9.5)
ys = [b[1] for b in boxes]
for i in range(len(ys) - 1):
    ax.add_patch(FancyArrowPatch((0.50, ys[i] - 0.062), (0.50, ys[i + 1] + 0.062),
                                 arrowstyle='-|>', mutation_scale=20, color='#333333', lw=1.5))
for _, ypos in excl:
    ax.add_patch(FancyArrowPatch((0.70, ypos), (0.74, ypos),
                                 arrowstyle='-|>', mutation_scale=15, color='#d62728', lw=1.2))
# 侧支箭头: 18,273 -> 16,462; 1,783 -> 1,504; 1,504 -> 789 (标注715)
ax.add_patch(FancyArrowPatch((0.30, 0.92 - 0.03), (0.26, 0.83), arrowstyle='-|>',
                             mutation_scale=15, color='#2ca02c', lw=1.2,
                             connectionstyle="arc3,rad=0.25"))
ax.add_patch(FancyArrowPatch((0.30, 0.56 - 0.03), (0.26, 0.47), arrowstyle='-|>',
                             mutation_scale=15, color='#2ca02c', lw=1.2,
                             connectionstyle="arc3,rad=0.25"))
ax.add_patch(FancyArrowPatch((0.135, 0.47 - 0.062), (0.32, 0.38 + 0.062), arrowstyle='-|>',
                             mutation_scale=15, color='#2ca02c', lw=1.2,
                             connectionstyle="arc3,rad=0.25"))
ax.text(0.16, 0.34, 'ICD-9-era HF\nexcluded (n = 715)', ha='center', va='center',
        fontsize=9, color='#2ca02c')
ax.add_patch(FancyArrowPatch((0.30, 0.38), (0.26, 0.20 + 0.062), arrowstyle='-|>',
                             mutation_scale=15, color='#2ca02c', lw=1.2,
                             connectionstyle="arc3,rad=0.18"))
ax.text(0.50, 0.085, 'Table 1: descriptive cohort n = 1,783 | Primary model, mortality and MIMIC mediation: new-onset cohort n = 789',
        ha='center', va='center', fontsize=9.5, color='#333333')
ax.set_xlim(0, 1)
ax.set_ylim(0.06, 1.0)
fig.tight_layout()
fig.savefig(f'{D}/figures/fig1_cohort_flow.png', dpi=200)
plt.close('all')
print('[done 116: fig1 v6 saved]')
