# -*- coding: utf-8 -*-
import warnings; warnings.filterwarnings('ignore')
import psycopg2, pandas as pd, numpy as np
conn = psycopg2.connect(dbname='eicu', host='127.0.0.1', port=5431,
                        user='postgres', password='123456')
ei = pd.read_sql('SELECT * FROM eicu_crd.temp_final_export', conn)
conn.close()
print('raw export:', len(ei))
ei['age_num'] = pd.to_numeric(ei['age'].astype(str).str.replace('> 89', '90'), errors='coerce')
ei = ei[(ei['age_num'] >= 18) & (ei['age_num'] <= 90) & (ei['los_icu_days'] >= 1)].copy()
print('after age/los filters:', len(ei))
tc = [c for c in ei.columns if c.startswith('troponin_i_') and c != 'troponin_i_first']
ei['trop_max_72h'] = ei[tc + ['troponin_i_first']].max(axis=1)
ei = ei[ei['trop_max_72h'].notna()].copy()
print('with troponin I:', len(ei))
ei['ne_used'] = (ei['norepi_max_rate'].fillna(0) > 0).astype(int)
ei['mi_proxy'] = ((ei['congestive_heart_failure'] == 0) & (ei['trop_max_72h'] > 0.04)).astype(int)
print('mi_proxy exposure n:', int(ei['mi_proxy'].sum()), '({:.1f}%)'.format(ei['mi_proxy'].mean() * 100))
print('NE used n:', int(ei['ne_used'].sum()))
print('28d mortality:', round(ei['mortality_28'].mean(), 3))
