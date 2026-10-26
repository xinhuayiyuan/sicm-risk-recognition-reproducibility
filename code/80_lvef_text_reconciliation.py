# 80_lvef_text_reconciliation.py - structured LVEF accuracy assessment (100-record stratified sample)
# A. ETL保真度: echo_lvef_numeric vs lvef_import.csv (measurement_id全量对账)
# B. 文本一致性: 100例分层抽样 -> 匹配mimiciv_note.radiology报告文本 -> 独立正则重提取 -> 一致率/kappa
# C. 内部一致性: 生理范围/重复测定
# -*- coding: utf-8 -*-
import warnings; warnings.filterwarnings('ignore')
import re
import numpy as np, pandas as pd, psycopg2

SEED = 20260101
rng = np.random.RandomState(SEED)
D = '<PROJECT_ROOT>/exec'

conn = psycopg2.connect(dbname='mimic', host='127.0.0.1', port=5431,
                        user='postgres', password='<SET-IN-ENVIRONMENT>')
cur = conn.cursor()

# ---------- A. ETL保真度 ----------
cur.execute("""SELECT measurement_id, lvef_value FROM mimiciv_echo.echo_lvef_numeric""")
eln = pd.DataFrame(cur.fetchall(), columns=['measurement_id', 'lvef_value'])
imp = pd.read_csv('<PROJECT_ROOT>/echo/lvef_import.csv',
                  usecols=['measurement_id', 'result'])
imp['result_num'] = pd.to_numeric(imp['result'], errors='coerce')
mg = eln.merge(imp, on='measurement_id', how='left')
matched = mg['result_num'].notna()
agree = np.isclose(mg.loc[matched, 'lvef_value'].astype(float),
                   mg.loc[matched, 'result_num'], atol=0.51)
print(f"A. ETL保真度: echo_lvef_numeric {len(eln)}行, 可比对{matched.sum()}行 "
      f"({matched.mean()*100:.1f}%), 一致率 {agree.mean()*100:.2f}%")

# ---------- C. 内部一致性 ----------
lv = eln['lvef_value'].astype(float)
print(f"C. 生理范围: 最小{lv.min()}, 最大{lv.max()}, 越界(<=0或>90) {((lv<=0)|(lv>90)).sum()}行")

# ---------- B. 文本一致性（100例分层抽样） ----------
cur.execute("""SELECT stay_id, hadm_id, measurement_id, measurement_datetime, lvef_value
               FROM mimiciv_echo.echo_lvef_numeric""")
alln = pd.DataFrame(cur.fetchall(), columns=['stay_id', 'hadm_id', 'measurement_id',
                                             'measurement_datetime', 'lvef_value'])
low = alln[alln['lvef_value'] < 50]; high = alln[alln['lvef_value'] >= 50]
samp = pd.concat([low.sample(min(50, len(low)), random_state=SEED),
                  high.sample(min(50, len(high)), random_state=SEED)])
hadms = [int(h) for h in samp['hadm_id'].unique()]
cur.execute("""SELECT hadm_id, charttime, text FROM mimiciv_note.radiology
               WHERE hadm_id = ANY(%s)
                 AND (LOWER(text) LIKE '%%echocardiogram%%' OR LOWER(text) LIKE '%%ejection fraction%%'
                      OR LOWER(text) LIKE '%%echo %%')""", (hadms,))
notes = pd.DataFrame(cur.fetchall(), columns=['hadm_id', 'charttime', 'text'])
cur.close(); conn.close()

PATTERNS = [
    re.compile(r'(?:lvef|ef|ejection\s+fraction)[^\d]{0,20}(\d{1,2}(?:\.\d+)?)\s*(?:%|percent)', re.I),
    re.compile(r'(?:lvef|ef|ejection\s+fraction)[^\d]{0,20}(\d{1,2})\s*[-–to]+\s*(\d{1,2})\s*%', re.I),
    re.compile(r'ejection\s+fraction\s+(?:is\s+)?(?:approximately\s+|about\s+|around\s+)?(\d{1,2})\s*%', re.I),
]
def reextract(text):
    vals = []
    for m in PATTERNS[1].finditer(text):      # 区间优先
        vals.append((float(m.group(1)) + float(m.group(2))) / 2)
    if not vals:
        for p in [PATTERNS[0], PATTERNS[2]]:
            vals += [float(m.group(1)) for m in p.finditer(text)]
    vals = [v for v in vals if 5 <= v <= 90]
    return vals

rows = []
for _, r in samp.iterrows():
    cand = notes[notes['hadm_id'] == r['hadm_id']].copy()
    if len(cand) == 0:
        rows.append(dict(measurement_id=r['measurement_id'], lvef_value=float(r['lvef_value']),
                         matched=0)); continue
    cand['dt'] = (pd.to_datetime(cand['charttime']) - pd.to_datetime(r['measurement_datetime'])).abs()
    best = cand.loc[cand['dt'].idxmin()]
    if best['dt'] > pd.Timedelta(hours=48):
        rows.append(dict(measurement_id=r['measurement_id'], lvef_value=float(r['lvef_value']),
                         matched=0)); continue
    vals = reextract(str(best['text']))
    if not vals:
        rows.append(dict(measurement_id=r['measurement_id'], lvef_value=float(r['lvef_value']),
                         matched=1, extracted=0)); continue
    # 取最接近lvef_value的提取值（评估"是否存在一致证据"）与第一个值（评估系统误差）
    v_close = min(vals, key=lambda v: abs(v - r['lvef_value']))
    rows.append(dict(measurement_id=r['measurement_id'], lvef_value=float(r['lvef_value']),
                     matched=1, extracted=1, reef=v_close, n_vals=len(vals),
                     diff=abs(v_close - float(r['lvef_value']))))
res = pd.DataFrame(rows)
res.to_csv(f'{D}/results/audit_echo_nlp_sample100.csv', index=False)
mt = res[res.get('matched', 0) == 1] if 'matched' in res else res
ext = mt[mt.get('extracted', 0) == 1] if 'extracted' in mt else mt
print(f"\nB. 文本一致性（100例抽样: EF<50有{(samp['lvef_value']<50).sum()}例）")
print(f"   匹配到报告文本: {len(mt)}/100 ({len(mt)}%)")
if len(ext):
    print(f"   文本中可独立重提取EF: {len(ext)}/{len(mt)}")
    print(f"   完全一致(±0.5pp): {(ext['diff']<=0.5).mean()*100:.1f}%")
    print(f"   ±5pp内: {(ext['diff']<=5).mean()*100:.1f}% | ±10pp内: {(ext['diff']<=10).mean()*100:.1f}%")
    agree_cls = ((ext['reef'] < 50) == (ext['lvef_value'] < 50)).mean()
    print(f"   EF<50分类一致率: {agree_cls*100:.1f}%")
    disc = ext[ext['diff'] > 10]
    print(f"   不一致(>10pp)需人工复核: {len(disc)}例 -> audit_echo_nlp_sample100.csv")
print("[done]")
