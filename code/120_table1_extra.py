# 120_table1_extra.py — Table 1 additional rows: peak troponin T and weight, by phenotype group
# -*- coding: utf-8 -*-
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, psycopg2
from scipy import stats

D = '<PROJECT_ROOT>/exec'
co = pd.read_csv(f'{D}/data/cohort_main.csv')

hf = (co['has_hf'] == 1) | (co['congestive_heart_failure'] == 1)
print('groups check:', len(co[~hf & (co['sicm'] == 1)]), len(co[~hf & (co['sicm'] == 0)]), len(co[hf]))
assert (len(co[~hf & (co['sicm'] == 1)]), len(co[~hf & (co['sicm'] == 0)]), len(co[hf])) == (215, 574, 994)

conn = psycopg2.connect(dbname='mimic', host='127.0.0.1', port=5431,
                        user='postgres', password='<SET-IN-ENVIRONMENT>')
wt = pd.read_sql("SELECT stay_id, weight FROM mimiciv_derived.first_day_weight", conn)
conn.close()
print('weight columns:', list(wt.columns), '| coverage in cohort:', round(co['stay_id'].isin(wt['stay_id']).mean(), 3))
co = co.merge(wt, on='stay_id', how='left')
g1 = co[~hf & (co['sicm'] == 1)]; g0 = co[~hf & (co['sicm'] == 0)]; g2 = co[hf]

def kw_p(a, b, c):
    return stats.kruskal(a.dropna(), b.dropna(), c.dropna()).pvalue

rows = []
for var, lab, nd in [('trop_max', 'Peak troponin T, ng/mL', 3), ('weight', 'Weight, kg', 1)]:
    p = kw_p(g1[var], g0[var], g2[var])
    nmiss = int(co[var].isna().sum())
    def mi(s):
        s = s.dropna()
        f = f"{{:.{nd}f}}".format
        return f"{f(s.median())} [{f(s.quantile(.25))}\u2013{f(s.quantile(.75))}]"
    rows.append([lab, mi(g1[var]), mi(g0[var]), mi(g2[var]),
                 f"{p:.3f}" if p >= 0.001 else "<0.001", nmiss])
    print(rows[-1])

pd.DataFrame(rows, columns=['var', 'g1', 'g0', 'g2', 'p', 'n_missing']).to_csv(
    f'{D}/results/v6_table1_extra_rows.csv', index=False)
print('[done 120]')
