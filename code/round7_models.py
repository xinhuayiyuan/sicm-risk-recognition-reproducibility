# -*- coding: utf-8 -*-
"""Round-7 HF-timing sensitivity and patient-level internal validation."""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "exec" / "data"
RESULTS = ROOT / "exec" / "results"
SEED = 20260101
co = pd.read_csv(DATA / "round7_cohort_timing.csv")
NUM = ["age","cci","sofa_24h","heart_rate_max","mbp_min","resp_rate_max","spo2_min",
       "temperature_max","lactate_max_24h","wbc_max","platelets_min","creatinine_max",
       "bun_max","bilirubin_total_max","sodium_min","potassium_max","glucose_max",
       "inr_max","ptt_max","bicarbonate_min","aniongap_max","hemoglobin_min",
       "ne_max_24h_strict","aki_stage_24h"]
COMORB = ["myocardial_infarct","peripheral_vascular_disease","cerebrovascular_disease",
          "dementia","chronic_pulmonary_disease","rheumatic_disease","mild_liver_disease",
          "diabetes_without_cc","diabetes_with_cc","renal_disease","malignant_cancer",
          "severe_liver_disease","metastatic_solid_tumor"]
BIN = ["gender_f","race_black","race_other"] + COMORB + ["vent_24h","rrt_firstday","lactate_miss","bili_miss"]
FEATS = NUM + BIN
CONF = ["age","sofa_24h","cci","lact_f","vent_24h","aki_stage_24h"]

def enrich(d):
    d = d.copy()
    rc = d.race.astype(str).str.upper()
    d["race_black"] = rc.str.contains("BLACK").astype(int)
    d["race_other"] = (~rc.str.contains("WHITE|BLACK")).astype(int)
    for c in COMORB + ["rrt_firstday"]:
        d[c] = d[c].fillna(0).astype(int)
    return d.reset_index(drop=True)

co = enrich(co)

def prep(a, b):
    med = a[NUM].median()
    xa, xb = a[FEATS].copy(), b[FEATS].copy()
    xa[NUM], xb[NUM] = xa[NUM].fillna(med), xb[NUM].fillna(med)
    scaler = StandardScaler().fit(xa[NUM])
    xa[NUM], xb[NUM] = scaler.transform(xa[NUM]), scaler.transform(xb[NUM])
    return xa, xb

def fit_grouped(a, y, groups):
    # Both penalty selection and refitting are confined to the development patients.
    inner = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=SEED)
    splits = list(inner.split(a, y, groups=groups))
    return LogisticRegressionCV(Cs=20, penalty="l1", solver="liblinear", cv=splits,
                                scoring="roc_auc", max_iter=3000,
                                random_state=SEED).fit(a, y)

def cal_stats(y, p):
    p = np.clip(p, 1e-6, 1-1e-6)
    lp = np.log(p/(1-p))
    intercept = np.asarray(sm.GLM(y, np.ones((len(y),1)), family=sm.families.Binomial(),
                                  offset=lp).fit().params)[0]
    slope = np.asarray(sm.Logit(y, sm.add_constant(lp)).fit(disp=0).params)[1]
    curve = calibration_curve(y, p, n_bins=10, strategy="quantile")
    return intercept, slope, float(np.mean(np.abs(curve[1]-curve[0])))

def grouped_bootstrap(d, y, p, seed, n=2000):
    rng = np.random.RandomState(seed)
    subjects = d.subject_id.to_numpy()
    unique = np.unique(subjects)
    values = []
    for _ in range(n):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([np.where(subjects == s)[0] for s in sampled])
        if len(np.unique(y[idx])) < 2:
            continue
        values.append(roc_auc_score(y[idx], p[idx]))
    return np.percentile(values, [2.5,97.5]).tolist()

def performance(d, p, label, seed):
    y = d.sicm.to_numpy()
    intercept, slope, ici = cal_stats(y, p)
    lo, hi = grouped_bootstrap(d, y, p, seed)
    return {"analysis":label,"n":len(d),"patients":d.subject_id.nunique(),"events":int(y.sum()),
            "AUC":roc_auc_score(y,p),"AUC_lo":lo,"AUC_hi":hi,
            "PR_AUC":average_precision_score(y,p),"Brier":brier_score_loss(y,p),
            "calibration_intercept":intercept,"calibration_slope":slope,"ICI":ici}

def outcome_models(d):
    x = sm.add_constant(pd.concat([d.sicm,d[CONF]],axis=1).astype(float))
    y = d.mort_28d.astype(float)
    groups = d.subject_id.to_numpy()
    adjusted = sm.GLM(y,x,family=sm.families.Binomial()).fit(
        cov_type="cluster",cov_kwds={"groups":groups})
    weighted = sm.GLM(y,x,family=sm.families.Binomial(),freq_weights=d.selection_weight).fit(
        cov_type="cluster",cov_kwds={"groups":groups})
    def estimate(model):
        ci = model.conf_int().loc["sicm"]
        return np.exp(model.params.sicm),np.exp(ci[0]),np.exp(ci[1]),model.pvalues.sicm
    adjusted_fit = sm.Logit(y,x).fit(disp=0)
    x0,x1 = x.copy(),x.copy()
    x0["sicm"],x1["sicm"] = 0.0,1.0
    rd = float((adjusted_fit.predict(x1)-adjusted_fit.predict(x0)).mean())
    return estimate(adjusted),estimate(weighted),rd

def rd_bootstrap(d,n_boot=500):
    rng = np.random.RandomState(SEED)
    subjects = d.subject_id.to_numpy()
    unique = np.unique(subjects)
    values = []
    for _ in range(n_boot):
        sampled = rng.choice(unique,size=len(unique),replace=True)
        parts = [d.iloc[np.where(subjects == s)[0]] for s in sampled]
        b = pd.concat(parts,ignore_index=True)
        x = sm.add_constant(pd.concat([b.sicm,b[CONF]],axis=1).astype(float),has_constant="add")
        try:
            fit = sm.Logit(b.mort_28d.astype(float),x).fit(disp=0)
            x0,x1 = x.copy(),x.copy()
            x0["sicm"],x1["sicm"] = 0.0,1.0
            values.append(float((fit.predict(x1)-fit.predict(x0)).mean()))
        except Exception:
            pass
    return np.percentile(values,[2.5,97.5]).tolist(),len(values)

def grouped_oof(d,label,seed_offset=0):
    y = d.sicm.to_numpy()
    oof = np.zeros(len(d))
    outer = StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED)
    for tr,va in outer.split(d,y,groups=d.subject_id):
        xa,xb = prep(d.iloc[tr],d.iloc[va])
        model = fit_grouped(xa,y[tr],d.iloc[tr].subject_id)
        oof[va] = model.predict_proba(xb)[:,1]
    return performance(d,oof,label,SEED+seed_offset)

# Patient-level holdout and OOF in the original 789-stay cohort.
# Scheme A is the current index-admission HF exclusion, not the strict-time cohort.
a = co[(co.has_hf.eq(0))&(co.congestive_heart_failure.eq(0))].copy().reset_index(drop=True)
y = a.sicm.to_numpy()
split = GroupShuffleSplit(n_splits=1,test_size=0.30,random_state=SEED)
tr,te = next(split.split(a,y,groups=a.subject_id))
xtr,xte = prep(a.iloc[tr],a.iloc[te])
model = fit_grouped(xtr,y[tr],a.iloc[tr].subject_id)
p_test = model.predict_proba(xte)[:,1]
validation = [performance(a.iloc[te],p_test,"patient_grouped_holdout",SEED+11),
              grouped_oof(a,"patient_grouped_oof",12)]
pd.DataFrame(validation).to_csv(RESULTS / "round7_patient_level_validation.csv",index=False)

schemes = {
    "A_current_index_admission_exclusion": a,
    "B_strict_time_exclude_uncertain": co[co.prior_adm_hf.eq(0)&co.index_hf.eq(0)].copy().reset_index(drop=True),
    "C_expanded_possible_new_onset": co[co.prior_adm_hf.eq(0)].copy().reset_index(drop=True),
}
association_rows = []
prediction_rows = []
for i,(name,d) in enumerate(schemes.items()):
    adj,ipw,rd = outcome_models(d)
    rdb,nok = rd_bootstrap(d)
    association_rows.append({
        "scheme":name,"n":len(d),"patients":d.subject_id.nunique(),"sicm_events":int(d.sicm.sum()),
        "deaths_sicm":int(d.loc[d.sicm.eq(1),"mort_28d"].sum()),
        "deaths_reference":int(d.loc[d.sicm.eq(0),"mort_28d"].sum()),
        "adjusted_OR":adj[0],"adjusted_OR_lo":adj[1],"adjusted_OR_hi":adj[2],"adjusted_p":adj[3],
        "IPW_OR":ipw[0],"IPW_OR_lo":ipw[1],"IPW_OR_hi":ipw[2],"IPW_p":ipw[3],
        "adjusted_RD":rd,"adjusted_RD_lo":rdb[0],"adjusted_RD_hi":rdb[1],"RD_bootstrap_ok":nok})
    prediction_rows.append(grouped_oof(d,name,100+i))
pd.DataFrame(association_rows).to_csv(RESULTS / "round7_hf_timing_association.csv",index=False)
pd.DataFrame(prediction_rows).to_csv(RESULTS / "round7_hf_timing_prediction.csv",index=False)
print(pd.DataFrame(association_rows).to_string(index=False))
print(pd.DataFrame(validation).to_string(index=False))
print(pd.DataFrame(prediction_rows).to_string(index=False))
print("[round7_models complete]")

