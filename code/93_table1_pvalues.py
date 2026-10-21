# 93_table1_pvalues.py - 表1补充组间比较P值（3组: 新发SICM / 参照 / 既存心脏病）
# 连续变量: Kruskal-Wallis; 分类变量: 卡方（期望频数<5用Fisher）
# -*- coding: utf-8 -*-
import numpy as np, pandas as pd
from scipy import stats

D = '<PROJECT_ROOT>/exec'
co = pd.read_csv(f'{D}/data/cohort_main.csv')
co['chf'] = co['congestive_heart_failure'].fillna(0)
co['grp'] = 'ref'
co.loc[(co['has_hf'] == 0) & (co['chf'] == 0) & (co['sicm'] == 1), 'grp'] = 'dn'
co.loc[~((co['has_hf'] == 0) & (co['chf'] == 0)), 'grp'] = 'pd'
groups = ['dn', 'ref', 'pd']
lab = {'dn': '新发SICM', 'ref': '参照(无HF史,非SICM)', 'pd': '既存心脏病(任意SICM状态)'}
co['grp_lab'] = co['grp'].map(lab)
co['gender_f'] = (co['gender'] == 'F').astype(int)

def p_cont(v):
    xs = [co.loc[co['grp'] == g, v].dropna() for g in groups]
    if any(len(x) < 5 for x in xs):
        return np.nan
    return stats.kruskal(*xs).pvalue

def p_cat(v):
    tab = pd.crosstab(co['grp'], co[v].fillna(0).astype(int))
    if (tab.values < 5).any():
        try:
            return stats.fisher_exact(tab)[1] if tab.shape == (3, 2) else stats.chi2_contingency(tab).pvalue
        except Exception:
            return stats.chi2_contingency(tab).pvalue
    return stats.chi2_contingency(tab).pvalue

def fmt_p(p):
    if pd.isna(p):
        return '—'
    return '<0.001' if p < 0.001 else f'{p:.3f}'

old = pd.read_csv(f'{D}/results/table1_era_spanning.csv')
cont_vars = ['age', 'sofa_24h', 'cci', 'heart_rate_max', 'mbp_min', 'lactate_max_24h',
             'creatinine_max', 'platelets_min', 'hemoglobin_min', 'min_ef', 'los_icu']
cat_vars = ['gender_f', 'myocardial_infarct', 'chronic_pulmonary_disease',
            'diabetes_without_cc', 'malignant_cancer', 'renal_disease', 'invasive_vent',
            'dobutamine', 'vaso_new_upgrade', 'hosp_mort', 'mort_28d']
var_map = dict(zip(
    ['年龄, 岁', 'SOFA-24h', 'Charlson指数', '心率max, /min', 'MBPmin, mmHg',
     '乳酸max, mmol/L', '肌酐max, mg/dL', '血小板min, 10^9/L', '血红蛋白min, g/dL',
     '最低LVEF, %', 'ICU住院, d',
     '女性', '心肌梗死史', '慢阻肺', '糖尿病', '恶性肿瘤', '慢性肾病', '有创通气',
     '多巴酚丁胺', '血管活性药升级', '住院死亡', '28天死亡'],
    cont_vars + cat_vars))

rows = []
for _, r in old.iterrows():
    v = r['变量']
    if v == 'n':
        rows.append(dict(r, P='—'))
        continue
    col = var_map.get(v)
    p = p_cont(col) if col in cont_vars else p_cat(col)
    rows.append(dict(r, P=fmt_p(p)))
out = pd.DataFrame(rows)
out.to_csv(f'{D}/results/table1_era_spanning_p.csv', index=False)
print(out.to_string(index=False))
print('[done]')
