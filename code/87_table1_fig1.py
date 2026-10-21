# 87_table1_fig1.py - 修正表型版表1 + 图1队列流程图
# -*- coding: utf-8 -*-
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

D = '<PROJECT_ROOT>/exec'
co = pd.read_csv(f'{D}/data/cohort_main.csv')

# 修正表型分组
co['chf'] = co['congestive_heart_failure'].fillna(0)
co['grp'] = '参照(无HF史,非SICM)'
co.loc[(co['has_hf'] == 0) & (co['chf'] == 0) & (co['sicm'] == 1), 'grp'] = '新发SICM'
co.loc[(co['has_hf'] == 0) & (co['chf'] == 0) & (co['sicm'] == 0), 'grp'] = '参照(无HF史,非SICM)'
co.loc[~((co['has_hf'] == 0) & (co['chf'] == 0)), 'grp'] = '既存心脏病(任意SICM状态)'
groups = ['新发SICM', '参照(无HF史,非SICM)', '既存心脏病(任意SICM状态)']

def cont(v):
    return {g: (lambda x: f"{x.median():.1f} [{x.quantile(.25):.1f}-{x.quantile(.75):.1f}]")(
        co.loc[co['grp'] == g, v].dropna()) for g in groups}

def cat(v):
    return {g: (lambda x: f"{int(x.sum())} ({x.mean()*100:.1f}%)")(
        co.loc[co['grp'] == g, v].fillna(0)) for g in groups}

co['gender_f'] = (co['gender'] == 'F').astype(int)
rows = [dict(变量='n', **{g: int((co['grp'] == g).sum()) for g in groups})]
for v, lab in [('age', '年龄, 岁'), ('sofa_24h', 'SOFA-24h'), ('cci', 'Charlson指数'),
               ('heart_rate_max', '心率max, /min'), ('mbp_min', 'MBPmin, mmHg'),
               ('lactate_max_24h', '乳酸max, mmol/L'), ('creatinine_max', '肌酐max, mg/dL'),
               ('platelets_min', '血小板min, 10^9/L'), ('hemoglobin_min', '血红蛋白min, g/dL'),
               ('min_ef', '最低LVEF, %'), ('los_icu', 'ICU住院, d')]:
    rows.append(dict(变量=lab, **cont(v)))
for v, lab in [('gender_f', '女性'), ('myocardial_infarct', '心肌梗死史'),
               ('chronic_pulmonary_disease', '慢阻肺'), ('diabetes_without_cc', '糖尿病'),
               ('malignant_cancer', '恶性肿瘤'), ('renal_disease', '慢性肾病'),
               ('invasive_vent', '有创通气'), ('dobutamine', '多巴酚丁胺'),
               ('vaso_new_upgrade', '血管活性药升级'), ('hosp_mort', '住院死亡'),
               ('mort_28d', '28天死亡')]:
    rows.append(dict(变量=lab, **cat(v)))
t1 = pd.DataFrame(rows)
t1.to_csv(f'{D}/results/table1_era_spanning.csv', index=False)
print(t1.to_string(index=False))

# ---- 图1 流程图 ----
fig, ax = plt.subplots(figsize=(8, 9)); ax.axis('off')
boxes = [
    ("Sepsis-3 ICU stays\n(MIMIC-IV 2008-2022)\nn = 18,273", 0.92),
    ("Evaluable (troponin + LVEF echo)\nn = 2,350", 0.74),
    ("Primary cohort (age 18-90, no ACS)\nn = 1,783", 0.56),
    ("De novo modelling cohort\n(no HF by ICD-9 428.x AND ICD-10 I50)\nn = 789", 0.38),
    ("De novo LV systolic SICM\nn = 215 (27.2%)", 0.20),
]
excl = [
    ("Excluded:\nno troponin or no echo (n=15,923)", 0.83),
    ("Excluded:\nage/ACS/prior cardiac surgery (n=567)", 0.65),
    ("HF history by either coding system\n(n=994, incl. 715 ICD-9-era CHF)", 0.47),
]
for (txt, ypos) in boxes:
    ax.add_patch(FancyBboxPatch((0.28, ypos - 0.055), 0.44, 0.11,
                                boxstyle="round,pad=0.01", fc='#eaf2fb', ec='#1f77b4'))
    ax.text(0.5, ypos, txt, ha='center', va='center', fontsize=10)
for (txt, ypos) in excl:
    ax.add_patch(FancyBboxPatch((0.76, ypos - 0.05), 0.22, 0.10,
                                boxstyle="round,pad=0.01", fc='#fdf0e8', ec='#d62728'))
    ax.text(0.87, ypos, txt, ha='center', va='center', fontsize=8)
ys = [b[1] for b in boxes]
for i in range(len(ys) - 1):
    ax.add_patch(FancyArrowPatch((0.5, ys[i] - 0.055), (0.5, ys[i + 1] + 0.055),
                                 arrowstyle='-|>', mutation_scale=18, color='#333333'))
for (txt, ypos) in excl:
    ax.add_patch(FancyArrowPatch((0.72, ypos), (0.76, ypos),
                                 arrowstyle='-|>', mutation_scale=14, color='#d62728'))
ax.set_xlim(0, 1); ax.set_ylim(0.1, 1.0)
fig.tight_layout(); fig.savefig(f'{D}/figures/fig1_cohort_flow.png', dpi=200)
print("\n[done] table1_era_spanning.csv + fig1_cohort_flow.png")
