# 64_landmark.py - V3.0: 72h landmark动态再分层（方案§6.3）
# landmark队列: 无HF, 72h内存活, 72h前未检出SICM
# 结局: 72h后新发SICM(EF<50首次出现于72-672h且cTn↑)
# 动态特征(24-72h窗口): 乳酸max/末次, NE最大速率; 比较 静态模型 vs +动态特征
# -*- coding: utf-8 -*-
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, psycopg2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

SEED = 20260101
rng = np.random.RandomState(SEED)
D = '<PROJECT_ROOT>/exec'
co = pd.read_csv(f'{D}/data/cohort_main.csv')
cr = pd.read_csv(f'{D}/data/comprisk_dn.csv')   # 含 t_ef50/t_death

conn = psycopg2.connect(dbname='mimic', host='127.0.0.1', port=5431,
                        user='postgres', password='<SET-IN-ENVIRONMENT>')
lac = pd.read_sql("""
    WITH w AS (
        SELECT i.stay_id, l.valuenum, l.charttime, i.intime
        FROM mimiciv_icu.icustays i
        JOIN mimiciv_hosp.labevents l ON l.hadm_id = i.hadm_id
        WHERE l.itemid IN (50813,52442,53154) AND l.valuenum IS NOT NULL
          AND l.charttime BETWEEN i.intime + INTERVAL '24 hours'
                              AND i.intime + INTERVAL '72 hours'
    )
    SELECT w.stay_id, MAX(w.valuenum) AS lac_max_24_72,
           (ARRAY_AGG(w.valuenum ORDER BY w.charttime DESC))[1] AS lac_last_24_72
    FROM w GROUP BY w.stay_id""", conn)
ne = pd.read_sql("""
    SELECT n.stay_id, MAX(n.vaso_rate) AS ne_max_24_72
    FROM mimiciv_derived.norepinephrine n
    JOIN mimiciv_icu.icustays i ON n.stay_id = i.stay_id
    WHERE n.starttime BETWEEN i.intime + INTERVAL '24 hours' AND i.intime + INTERVAL '72 hours'
    GROUP BY n.stay_id""", conn)
conn.close()

dn = co[co['has_hf'] == 0].merge(cr[['stay_id', 't_ef50', 't_death', 'status']], on='stay_id')
dn = dn.merge(lac, on='stay_id', how='left').merge(ne, on='stay_id', how='left')

# landmark队列: 72h内存活(或无死亡记录), 72h前无SICM
alive72 = dn['t_death'].isna() | (dn['t_death'] > 72)
no_sicm72 = dn['t_ef50'].isna() | (dn['t_ef50'] > 72)
lm = dn[alive72 & no_sicm72].copy()
# 结局: 72h后SICM
lm['y_post'] = ((lm['status'] == 1) & (lm['t_ef50'] > 72)).astype(int)
print(f"landmark队列 n={len(lm)}, 72h后SICM={lm['y_post'].sum()} ({lm['y_post'].mean()*100:.1f}%)")

NUM = ['age','cci','sofa_24h','heart_rate_max','mbp_min','resp_rate_max','spo2_min',
       'lactate_max_24h','wbc_max','platelets_min','creatinine_max','hemoglobin_min']
DYN = ['lac_max_24_72', 'lac_last_24_72', 'ne_max_24_72']
lm['ne_max_24_72'] = lm['ne_max_24_72'].fillna(0)
lm['lac_clearance'] = (lm['lactate_max_24h'] - lm['lac_last_24_72']) / lm['lactate_max_24h']
DYN.append('lac_clearance')

def run_lm(feats, tag):
    dd = lm.dropna(subset=['y_post'])
    X = dd[feats].copy()
    for c in feats:
        X[c] = X[c].fillna(X[c].median())
    yy = dd['y_post'].values
    if yy.sum() < 20:
        print(f"[{tag}] 事件不足({yy.sum()}), 跳过"); return None
    tr, te = train_test_split(np.arange(len(dd)), test_size=0.3, stratify=yy, random_state=SEED)
    sc = StandardScaler().fit(X.iloc[tr])
    m = LogisticRegressionCV(Cs=10, penalty='l1', solver='liblinear', cv=5,
                             scoring='roc_auc', max_iter=3000, random_state=SEED)
    m.fit(sc.transform(X.iloc[tr]), yy[tr])
    p = m.predict_proba(sc.transform(X.iloc[te]))[:, 1]
    aucs = [roc_auc_score(yy[te][i], p[i]) for i in
            [rng.randint(0, len(te), len(te)) for _ in range(500)]
            if len(np.unique(yy[te][i])) > 1]
    ci = np.percentile(aucs, [2.5, 97.5])
    auc = roc_auc_score(yy[te], p)
    print(f"[{tag}] AUC={auc:.3f} (95%CI {ci[0]:.3f}-{ci[1]:.3f}), n_test={len(te)}, 事件={yy[te].sum()}")
    return dict(模型=tag, AUC=round(auc, 3), lo=round(ci[0], 3), hi=round(ci[1], 3))

r1 = run_lm(NUM, 'landmark静态(24h基线)')
r2 = run_lm(NUM + DYN, 'landmark+24-72h动态')
pd.DataFrame([r for r in [r1, r2] if r]).to_csv(f'{D}/results/v3_landmark72.csv', index=False)
print("[done]")
