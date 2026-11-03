# -*- coding: utf-8 -*-
"""Current-cohort, current-adjustment, patient-cluster full-procedure IPW bootstrap."""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/"exec"/"data";RESULTS=ROOT/"exec"/"results"
SEED=20260101;rng=np.random.RandomState(SEED)
full=pd.read_csv(DATA/"round7_selection_source.csv")
co=pd.read_csv(DATA/"round7_cohort_timing.csv")
dn=co[(co.has_hf==0)&(co.congestive_heart_failure==0)].copy()
dn["lact_f"]=dn.lactate_max_24h.fillna(dn.lactate_max_24h.median())
CONF=["age","sofa_24h","cci","lact_f","vent_24h","aki_stage_24h"]
PSF=["age","sofa_24h","heart_rate_max","mbp_min","resp_rate_max","spo2_min",
     "lactate_max_24h","wbc_max","creatinine_max","cci"]

def ps_weights(frame):
    x=frame[PSF].copy()
    for c in PSF: x[c]=x[c].fillna(x[c].median())
    x["invasive_vent"]=frame.invasive_vent.fillna(0)
    x["gender_f"]=frame.gender.eq("F").astype(int)
    sc=StandardScaler().fit(x)
    model=LogisticRegression(max_iter=1500).fit(sc.transform(x),frame.evaluable)
    ps=np.clip(model.predict_proba(sc.transform(x))[:,1],0.02,0.98)
    prev=frame.evaluable.mean();w=np.where(frame.evaluable.eq(1),prev/ps,1.0)
    we=w[frame.evaluable.eq(1)].copy();lo,hi=np.percentile(we,[1,99])
    w[frame.evaluable.eq(1)]=np.clip(we,lo,hi)
    return np.asarray(w)

def cluster_sample(frame,rng):
    indices=frame.groupby('subject_id').indices
    unique=np.array(list(indices))
    sampled=rng.choice(unique,size=len(unique),replace=True)
    idx=np.concatenate([indices[s] for s in sampled])
    return frame.iloc[idx].reset_index(drop=True)

full["sw"]=ps_weights(full)
def smd(x,g,w=None):
    x1,x0=x[g==1].astype(float),x[g==0].astype(float)
    if w is None: m1,m0,v1,v0=x1.mean(),x0.mean(),x1.var(),x0.var()
    else:
        w1,w0=w[g==1],w[g==0]
        m1,m0=np.average(x1,weights=w1),np.average(x0,weights=w0)
        v1=np.average((x1-m1)**2,weights=w1);v0=np.average((x0-m0)**2,weights=w0)
    sp=np.sqrt((v1+v0)/2)
    return (m1-m0)/sp if sp>0 else 0.0
x=full[PSF].copy()
for c in PSF: x[c]=x[c].fillna(x[c].median())
x["invasive_vent"]=full.invasive_vent.fillna(0);x["gender_f"]=full.gender.eq("F").astype(int)
g=full.evaluable.to_numpy()
rows=[{"covariate":c,"SMD_unweighted":smd(x[c].to_numpy(),g),
       "SMD_weighted":smd(x[c].to_numpy(),g,full.sw.to_numpy())} for c in x.columns]
pd.DataFrame(rows).to_csv(RESULTS/"round7_ipw_balance_smd.csv",index=False)

ors=[]
for b in range(300):
    try:
        fb=cluster_sample(full,rng);fb["sw"]=ps_weights(fb)
        db=cluster_sample(dn,rng).drop(columns=["selection_weight"],errors="ignore")
        db=db.merge(fb[["stay_id","sw"]].groupby("stay_id").first().reset_index(),on="stay_id",how="left")
        db=db.dropna(subset=["sw","mort_28d","sicm"]+CONF);db=db[db.sw>0]
        X=sm.add_constant(pd.concat([db.sicm,db[CONF]],axis=1).astype(float))
        fit=sm.GLM(db.mort_28d.astype(float),X,family=sm.families.Binomial(),freq_weights=db.sw).fit()
        ors.append(np.exp(fit.params.sicm))
    except Exception:
        continue
lo,hi=np.percentile(ors,[2.5,97.5])
out=pd.DataFrame({"analysis":["patient_cluster_full_procedure_ipw"],"replicates":[300],"successful":[len(ors)],
                  "OR_lo":[lo],"OR_hi":[hi]})
out.to_csv(RESULTS/"round7_ipw_full_bootstrap.csv",index=False)
print(out.to_string(index=False));print(pd.DataFrame(rows).to_string(index=False))
