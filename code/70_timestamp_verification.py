# 70_timestamp_verification.py - 核查: "24h预测因子"的真实时间窗（数据泄漏核查）
# ①post_max_ned 与 0-24h / 全病程 NE最大速率的一致性
# ②dobutamine/milrinone/invasive_vent 使用者在24h后才启用的比例
# -*- coding: utf-8 -*-
import pandas as pd, numpy as np, psycopg2

D = '<PROJECT_ROOT>/exec'
co = pd.read_csv(f'{D}/data/cohort_main.csv')
dn = co[co['has_hf'] == 0]
ids = [int(s) for s in dn['stay_id']]

conn = psycopg2.connect(dbname='mimic', host='127.0.0.1', port=5431,
                        user='postgres', password='<SET-IN-ENVIRONMENT>')
cur = conn.cursor()

# ① NE: 0-24h max vs 全病程max
cur.execute("""
    SELECT n.stay_id,
           MAX(n.vaso_rate) FILTER (WHERE n.starttime <= i.intime + INTERVAL '24 hours') AS ne_max_24h,
           MAX(n.vaso_rate) AS ne_max_all,
           MIN(n.starttime) AS first_ne_start, i.intime
    FROM mimiciv_derived.norepinephrine n
    JOIN mimiciv_icu.icustays i ON n.stay_id = i.stay_id
    WHERE n.stay_id = ANY(%s) GROUP BY n.stay_id, i.intime""", (ids,))
ne = pd.DataFrame(cur.fetchall(), columns=['stay_id', 'ne_max_24h', 'ne_max_all', 'first_ne_start', 'intime'])
ne['ne_start_h'] = (pd.to_datetime(ne['first_ne_start']) - pd.to_datetime(ne['intime'])).dt.total_seconds() / 3600

# ② dobutamine / milrinone / ventilation 首次启用时间
out = {}
for tab, col in [('dobutamine', 'starttime'), ('milrinone', 'starttime')]:
    cur.execute(f"""
        SELECT t.stay_id, MIN(EXTRACT(EPOCH FROM (t.{col} - i.intime))/3600.0) AS start_h
        FROM mimiciv_derived.{tab} t JOIN mimiciv_icu.icustays i ON t.stay_id = i.stay_id
        WHERE t.stay_id = ANY(%s) GROUP BY t.stay_id""", (ids,))
    out[tab] = pd.DataFrame(cur.fetchall(), columns=['stay_id', 'start_h'])
cur.execute("""
    SELECT v.stay_id, MIN(EXTRACT(EPOCH FROM (v.starttime - i.intime))/3600.0) AS start_h
    FROM mimiciv_derived.ventilation v JOIN mimiciv_icu.icustays i ON v.stay_id = i.stay_id
    WHERE v.stay_id = ANY(%s) AND v.ventilation_status ILIKE '%%Invasive%%'
    GROUP BY v.stay_id""", (ids,))
out['vent'] = pd.DataFrame(cur.fetchall(), columns=['stay_id', 'start_h'])
cur.close(); conn.close()

chk = dn[['stay_id', 'post_max_ned']].merge(ne, on='stay_id', how='left')
print("=== ① post_max_ned 时间窗核查（有NE记录者 n=%d） ===" % chk['ne_max_all'].notna().sum())
sub = chk.dropna(subset=['ne_max_all'])
print(f"post_max_ned == 全病程max 完全一致比例: {(np.isclose(sub['post_max_ned'], sub['ne_max_all'])).mean():.3f}")
print(f"post_max_ned == 0-24h max 完全一致比例: {(np.isclose(sub['post_max_ned'], sub['ne_max_24h'].fillna(-9))).mean():.3f}")
print(f"post_max_ned 与 全病程max 相关系数: {sub['post_max_ned'].corr(sub['ne_max_all']):.3f}")
print(f"post_max_ned 与 0-24h max  相关系数: {sub['post_max_ned'].corr(sub['ne_max_24h']):.3f}")
print(f"post_max_ned > 0-24h max (即含24h后信息) 的比例: {(sub['post_max_ned'] > sub['ne_max_24h'].fillna(0) + 1e-9).mean():.3f}")
print(f"NE首次启用>24h者占有NE记录者比例: {(ne['ne_start_h'] > 24).mean():.3f}")

print("\n=== ② 治疗变量24h后启用比例（占各变量使用者） ===")
for k, dd in out.items():
    if len(dd):
        print(f"{k}: 使用者n={len(dd)}, 24h后才启用={(dd['start_h'] > 24).mean() * 100:.1f}%, "
              f"中位启用时间={dd['start_h'].median():.1f}h")
chk.to_csv(f'{D}/results/audit_ne_timing.csv', index=False)
print("\n[done]")
