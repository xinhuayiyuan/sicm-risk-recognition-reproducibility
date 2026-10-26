# 104_eligibility_sensitivity.py — 未执行纳入标准（first ICU stay / los>=1d）的影响评估
# -*- coding: utf-8 -*-
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

SEED = 20260101
D = '<PROJECT_ROOT>/exec'
co = pd.read_csv(f'{D}/data/cohort_main.csv')
aki = pd.read_csv(f'{D}/data/aki_stage_24h.csv')
st = pd.read_csv(f'{D}/data/cohort_dn_strict24h.csv',
                 usecols=['stay_id', 'ne_max_24h_strict', 'dobutamine_24h',
                          'milrinone_24h', 'vent_24h'])
co = co.merge(aki, on='stay_id', how='left').merge(st, on='stay_id', how='left')
co['aki_stage_24h'] = co['aki_stage_24h'].fillna(0).astype(int)

non_first = co['first_icu_stay'] == False
short = co['los_icu'] < 1
print(f'主队列 n=1783: 非首次stay {non_first.sum()} (其中SICM {co.loc[non_first,"sicm"].sum()}), '
      f'<24h stay {short.sum()} (其中SICM {co.loc[short,"sicm"].sum()}), '
      f'并集 {(non_first|short).sum()} (SICM {co.loc[non_first|short,"sicm"].sum()})')

dn = co[(co['has_hf'] == 0) & (co['congestive_heart_failure'] == 0)].copy().reset_index(drop=True)
dn['lact_f'] = dn['lactate_max_24h'].fillna(dn['lactate_max_24h'].median())
dn['vent_24h'] = dn['vent_24h'].fillna(0).astype(int)
CONF = ['age', 'sofa_24h', 'cci', 'lact_f', 'vent_24h', 'aki_stage_24h']

def quick(dsub, tag):
    d = dsub.copy().reset_index(drop=True)
    X2 = sm.add_constant(pd.concat([d['sicm'], d[CONF]], axis=1).astype(float))
    m = sm.Logit(d['mort_28d'].astype(float), X2).fit(disp=0)
    ci = np.exp(m.conf_int().loc['sicm'])
    NUM = ['age','cci','sofa_24h','heart_rate_max','mbp_min','resp_rate_max','spo2_min',
           'temperature_max','lactate_max_24h','wbc_max','platelets_min','creatinine_max',
           'bun_max','bilirubin_total_max','sodium_min','potassium_max','glucose_max',
           'inr_max','ptt_max','bicarbonate_min','aniongap_max','hemoglobin_min',
           'ne_max_24h_strict','aki_stage_24h']
    COMORB = ['myocardial_infarct','peripheral_vascular_disease','cerebrovascular_disease',
              'dementia','chronic_pulmonary_disease','rheumatic_disease','mild_liver_disease',
              'diabetes_without_cc','diabetes_with_cc','renal_disease','malignant_cancer',
              'severe_liver_disease','metastatic_solid_tumor','aids']
    BIN = ['gender_f','race_black','race_other'] + COMORB + \
          ['vent_24h','dobutamine_24h','milrinone_24h','rrt_firstday','lactate_miss','bili_miss']
    d['gender_f'] = (d['gender'] == 'F').astype(int)
    rc = d['race'].astype(str).str.upper()
    d['race_black'] = rc.str.contains('BLACK').astype(int)
    d['race_other'] = (~rc.str.contains('WHITE|BLACK')).astype(int)
    for c in COMORB + ['rrt_firstday']:
        d[c] = d[c].fillna(0).astype(int)
    for c in ['dobutamine_24h','milrinone_24h']:
        d[c] = d[c].fillna(0).astype(int)
    FEATS = [f for f in (NUM + BIN) if f != 'aids']
    y = d['sicm'].values
    tr, te = train_test_split(d.index, test_size=0.3, stratify=y, random_state=SEED)
    dtr, dte = d.loc[tr], d.loc[te]
    med = dtr[NUM].median()
    Xtr = dtr[FEATS].copy(); Xte = dte[FEATS].copy()
    Xtr[NUM] = Xtr[NUM].fillna(med); Xte[NUM] = Xte[NUM].fillna(med)
    sc = StandardScaler().fit(Xtr[NUM])
    Xtr[NUM] = sc.transform(Xtr[NUM]); Xte[NUM] = sc.transform(Xte[NUM])
    las = LogisticRegressionCV(Cs=20, penalty='l1', solver='liblinear', cv=10,
                               scoring='roc_auc', max_iter=3000, random_state=SEED).fit(Xtr, y[tr])
    auc = roc_auc_score(y[te], las.predict_proba(Xte)[:, 1])
    print(f'{tag}: n={len(d)}, SICM={int(d.sicm.sum())} ({d.sicm.mean()*100:.1f}%), '
          f'死亡 {d.loc[d.sicm==1,"mort_28d"].mean()*100:.1f}% vs {d.loc[d.sicm==0,"mort_28d"].mean()*100:.1f}% | '
          f'调整OR={np.exp(m.params["sicm"]):.3f} ({ci[0]:.3f}-{ci[1]:.3f}) P={m.pvalues["sicm"]:.4f} | '
          f'测试集AUC={auc:.3f}')

quick(dn, '全量修正队列(现稿)')
quick(dn[~((dn['first_icu_stay'] == False) | (dn['los_icu'] < 1))], '执行两项标准后')
quick(dn[dn['first_icu_stay'] != False], '仅执行first-stay')
quick(dn[dn['los_icu'] >= 1], '仅执行los>=1d')
