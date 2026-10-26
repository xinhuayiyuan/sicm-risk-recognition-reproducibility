# 10_build_final_dataset.py
# 阶段B-步骤1：锁定主分析队列并构建建模数据集（方案V2.0 §四）
# 主分析队列: evaluable=1, age 18-90, 排ACS; 附 anchor_year_group
# 输出: <PROJECT_ROOT>/exec/data/cohort_main.csv
# -*- coding: utf-8 -*-
import os
import pandas as pd
import psycopg2

os.makedirs('<PROJECT_ROOT>/exec/data', exist_ok=True)
os.makedirs('<PROJECT_ROOT>/exec/results', exist_ok=True)
os.makedirs('<PROJECT_ROOT>/exec/figures', exist_ok=True)

conn = psycopg2.connect(dbname='mimic', host='127.0.0.1', port=5431,
                        user='postgres', password='<SET-IN-ENVIRONMENT>')
df = pd.read_sql("""
    SELECT d.*, p.anchor_year_group
    FROM mimiciv_echo.sicm_analysis_dataset d
    JOIN mimiciv_hosp.patients p ON d.subject_id = p.subject_id
    WHERE d.evaluable = 1 AND d.age >= 18 AND d.age <= 90 AND d.has_acs = 0
""", conn)
conn.close()

# ---- 结局定义（方案4.4节） ----
df['sicm'] = ((df['ef_criterion'] == 1) & (df['trop_gt014'] == 1)).astype(int)
# 敏感性结局 S1a-d
df['sicm_ef40'] = ((df['min_ef'] < 40) & (df['trop_gt014'] == 1)).astype(int)
df['sicm_drop15'] = (((df['min_ef'] < 50) | (df['max_ef_drop'] >= 15)) & (df['trop_gt014'] == 1)).astype(int)
df['sicm_img'] = (df['ef_criterion'] == 1).astype(int)                      # 纯影像
df['sicm_defB'] = ((df['ef_criterion'] == 1) & (df['trop_gt014'] == 1)
                   & (df['vaso_new_upgrade'] == 1)).astype(int)             # 复合定义B

# ---- 表型分层 ----
df['phenotype'] = 'non_sicm'
df.loc[(df['sicm'] == 1) & (df['has_hf'] == 0), 'phenotype'] = 'de_novo'
df.loc[(df['sicm'] == 1) & (df['has_hf'] == 1), 'phenotype'] = 'preexist'
# 严格新发（临床病史法: 无HF/MI/ACS史）
df['denovo_strict'] = ((df['sicm'] == 1) & (df['has_hf'] == 0)
                       & (df['myocardial_infarct'] == 0)).astype(int)

# ---- 亚组变量（方案6.1节） ----
df['age_grp'] = pd.cut(df['age'], [0, 65, 75, 200], labels=['<65', '65-74', '>=75'])
df['sofa_grp'] = (df['sofa_24h'] >= 8).astype(int)
df['shock_24h'] = (df['post_max_ned'] > 0).astype(int)   # 24h内已用NE（休克代理）
df['aki_grp'] = df['aki_stage_max'].fillna(0).astype(int)

# ---- 缺失指示变量（乳酸/胆红素） ----
df['lactate_miss'] = df['lactate_max_24h'].isna().astype(int)
df['bili_miss'] = df['bilirubin_total_max'].isna().astype(int)

df.to_csv('<PROJECT_ROOT>/exec/data/cohort_main.csv', index=False)

# ---- 队列流水账（Figure 1用） ----
print("=== 队列锁定（方案V2.0 主分析队列） ===")
print(f"主分析队列 n = {len(df)}")
print(f"  新发SICM(de_novo)      : {(df.phenotype=='de_novo').sum()}")
print(f"  既存心脏病SICM(preexist): {(df.phenotype=='preexist').sum()}")
print(f"  非SICM                 : {(df.phenotype=='non_sicm').sum()}")
print(f"  28天死亡: {df.mort_28d.mean():.3f}")
print("\n新发建模队列 (has_hf=0):")
dn = df[df['has_hf'] == 0]
print(f"  n={len(dn)}, 事件={dn.sicm.sum()}, 事件率={dn.sicm.mean():.3f}")
print("既存心脏病队列 (has_hf=1):")
pd_ = df[df['has_hf'] == 1]
print(f"  n={len(pd_)}, 事件={pd_.sicm.sum()}, 事件率={pd_.sicm.mean():.3f}")
print("\n敏感性结局事件数:")
for c in ['sicm_ef40', 'sicm_drop15', 'sicm_img', 'sicm_defB']:
    print(f"  {c}: {df.loc[df.has_hf==0, c].sum()} (新发队列)")
print("\n年代分组:")
print(df.groupby('anchor_year_group').agg(n=('sicm', 'size'), denovo_events=('sicm', lambda s: s[df.loc[s.index, 'has_hf'] == 0].sum())))
print("\n[done] cohort_main.csv saved")
