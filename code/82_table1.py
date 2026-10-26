# 82_table1.py - 表1: 基线特征（按表型分层）
# -*- coding: utf-8 -*-
import numpy as np, pandas as pd

D = '<PROJECT_ROOT>/exec'
df = pd.read_csv(f'{D}/data/cohort_main.csv')
grp_order = ['de_novo', 'preexist', 'non_sicm']
grp_lab = {'de_novo': '新发SICM', 'preexist': '既存心脏病SICM', 'non_sicm': '非SICM'}
df['grp'] = df['phenotype'].map(grp_lab)

def cont(v):
    out = {}
    for g in grp_order:
        x = df.loc[df['phenotype'] == g, v].dropna()
        out[grp_lab[g]] = f"{x.median():.1f} [{x.quantile(.25):.1f}-{x.quantile(.75):.1f}]"
    return out

def cat(v):
    out = {}
    for g in grp_order:
        x = df.loc[df['phenotype'] == g, v].fillna(0)
        out[grp_lab[g]] = f"{int(x.sum())} ({x.mean()*100:.1f}%)"
    return out

rows = []
rows.append(dict(变量='n', **{grp_lab[g]: int((df['phenotype']==g).sum()) for g in grp_order}))
for v, lab in [('age', '年龄, 岁'), ('sofa_24h', 'SOFA-24h'), ('cci', 'Charlson指数'),
               ('heart_rate_max', '心率max'), ('mbp_min', 'MBPmin'),
               ('lactate_max_24h', '乳酸max'), ('creatinine_max', '肌酐max'),
               ('bilirubin_total_max', '胆红素max'), ('platelets_min', '血小板min'),
               ('hemoglobin_min', '血红蛋白min'), ('min_ef', '最低LVEF, %'),
               ('los_icu', 'ICU住院, d'), ('los_hospital', '住院, d')]:
    rows.append(dict(变量=lab, **cont(v)))
for v, lab in [('gender_f', '女性'), ('myocardial_infarct', '心肌梗死史'),
               ('congestive_heart_failure', '心衰史'), ('chronic_pulmonary_disease', '慢阻肺'),
               ('diabetes_without_cc', '糖尿病'), ('malignant_cancer', '恶性肿瘤'),
               ('severe_liver_disease', '重度肝病'), ('renal_disease', '慢性肾病'),
               ('invasive_vent', '有创通气'), ('dobutamine', '多巴酚丁胺'),
               ('rrt_any', '肾脏替代'), ('vaso_new_upgrade', '血管活性药升级'),
               ('hosp_mort', '住院死亡'), ('mort_28d', '28天死亡')]:
    df['gender_f'] = (df['gender'] == 'F').astype(int)
    rows.append(dict(变量=lab, **cat(v)))

t1 = pd.DataFrame(rows)
t1.to_csv(f'{D}/results/table1_baseline_by_phenotype.csv', index=False)
print(t1.to_string(index=False))
print("\n[done]")
