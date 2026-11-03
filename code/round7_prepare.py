# -*- coding: utf-8 -*-
"""Round-7 corrected upstream cohort. Requires MIMIC_PASSWORD; local record-level output."""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import psycopg2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"exec"/"data";RESULTS=ROOT/"exec"/"results"
co=pd.read_csv(DATA/"cohort_main.csv")
old_timing=pd.read_csv(DATA/"round6_cohort_timing.csv")
conn=psycopg2.connect(dbname="mimic",host="127.0.0.1",port=5431,user="postgres",
                      password=os.environ["MIMIC_PASSWORD"],connect_timeout=10)
conn.cursor().execute("SET statement_timeout='120s'")
all_stays=pd.read_sql("SELECT stay_id,hadm_id FROM mimiciv_echo.sicm_analysis_dataset",conn)
hadms=[int(x) for x in all_stays.hadm_id.unique()]
acs=pd.read_sql("""SELECT DISTINCT hadm_id FROM mimiciv_hosp.diagnoses_icd
WHERE hadm_id=ANY(%s) AND ((icd_version=9 AND (icd_code LIKE '410%%' OR icd_code LIKE '411%%'))
OR (icd_version=10 AND (icd_code LIKE 'I21%%' OR icd_code LIKE 'I22%%')))""",conn,params=(hadms,))
all_stays["has_acs_corrected"]=all_stays.hadm_id.isin(acs.hadm_id).astype(int)
co=co.drop(columns=["has_acs"],errors="ignore").merge(all_stays,on=["stay_id","hadm_id"],validate="one_to_one")
co["has_acs"]=co.has_acs_corrected
co=co.drop(columns=["has_acs_corrected"])
timing_cols=["stay_id","prior_adm_hf","index_hf","hf_time_group","sepsis_h","deathtime_h","dod_h","trop_positive_h","death_h"]
co=co.merge(old_timing[timing_cols],on="stay_id",how="left",validate="one_to_one")

# Qualifying low LVEF and decline follow-up must occur at/after suspected infection.
ids=[int(x) for x in co.stay_id]
post_ef=pd.read_sql("""SELECT e.stay_id,MIN(e.lvef_value) AS min_ef_post,
COUNT(DISTINCT e.measurement_id) AS n_echo_post,
MIN(CASE WHEN e.lvef_value<50 THEN e.hours_from_icu_intime END) AS ef50_post_h
FROM mimiciv_echo.echo_lvef_numeric e JOIN mimiciv_echo.sepsis_icu_cohort s ON e.stay_id=s.stay_id
WHERE e.stay_id=ANY(%s) AND e.measurement_datetime>=s.suspected_infection_time
GROUP BY e.stay_id""",conn,params=(ids,))
post_drop=pd.read_sql("""SELECT f.stay_id,MAX(f.lvef_drop_absolute) AS max_ef_drop_post,
MIN(CASE WHEN f.lvef_drop_absolute>=10 THEN f.followup_hours END) AS decline_post_h
FROM mimiciv_echo.echo_lvef_followup f JOIN mimiciv_echo.sepsis_icu_cohort s ON f.stay_id=s.stay_id
WHERE f.stay_id=ANY(%s) AND f.followup_datetime>=s.suspected_infection_time
GROUP BY f.stay_id""",conn,params=(ids,))
co=co.merge(post_ef,on="stay_id",how="left",validate="one_to_one")
co=co.merge(post_drop,on="stay_id",how="left",validate="one_to_one")
co["min_ef_original"]=co.min_ef
co["max_ef_drop_original"]=co.max_ef_drop
co["min_ef"]=co.min_ef_post
co["max_ef_drop"]=co.max_ef_drop_post
co["ef_criterion"]=((co.min_ef<50)|(co.max_ef_drop>=10)).astype(int)
co["sicm"]=((co.ef_criterion==1)&(co.trop_gt014==1)).astype(int)
co["sicm_ef40"]=((co.min_ef<40)&(co.trop_gt014==1)).astype(int)
co["sicm_drop15"]=(((co.min_ef<50)|(co.max_ef_drop>=15))&(co.trop_gt014==1)).astype(int)
co["sicm_img"]=(co.ef_criterion==1).astype(int)
co["sicm_defB"]=((co.sicm==1)&(co.vaso_new_upgrade==1)).astype(int)
co["phenotype"]="non_sicm"
co.loc[(co.sicm==1)&(co.has_hf==0),"phenotype"]="de_novo"
co.loc[(co.sicm==1)&(co.has_hf==1),"phenotype"]="preexist"
co["denovo_strict"]=((co.sicm==1)&(co.has_hf==0)&(co.myocardial_infarct==0)).astype(int)
co["ef50_h"]=co.ef50_post_h
co["decline_h"]=co.decline_post_h
co["imaging_h"]=co.ef50_h
co.loc[co.sicm.eq(1)&co.imaging_h.isna(),"imaging_h"]=co.decline_h
co["sicm_onset_h"]=np.nan
mask=co.sicm.eq(1)
co.loc[mask,"sicm_onset_h"]=co.loc[mask,["sepsis_h","imaging_h","trop_positive_h"]].max(axis=1)

# Strict starts within ICU 0-24 h; no pre-ICU start times are admitted.
ne=pd.read_sql("""SELECT n.stay_id,MAX(n.vaso_rate) AS ne_max_24h_corrected
FROM mimiciv_derived.norepinephrine n JOIN mimiciv_icu.icustays i ON n.stay_id=i.stay_id
WHERE n.stay_id=ANY(%s) AND n.starttime>=i.intime AND n.starttime<=i.intime+INTERVAL '24 hours'
GROUP BY n.stay_id""",conn,params=(ids,))
co=co.merge(ne,on="stay_id",how="left",validate="one_to_one")
co["ne_max_24h_strict"]=co.ne_max_24h_corrected.fillna(0)
for tab in ["dobutamine","milrinone","ventilation"]:
    condition="AND ventilation_status ILIKE '%%Invasive%%'" if tab=="ventilation" else ""
    flag=pd.read_sql(f"""SELECT DISTINCT t.stay_id FROM mimiciv_derived.{tab} t
    JOIN mimiciv_icu.icustays i ON t.stay_id=i.stay_id WHERE t.stay_id=ANY(%s)
    AND t.starttime>=i.intime AND t.starttime<=i.intime+INTERVAL '24 hours' {condition}""",conn,params=(ids,))
    col="vent_24h" if tab=="ventilation" else tab+"_24h"
    co[col]=co.stay_id.isin(flag.stay_id).astype(int)

# Corrected ACS is also applied to the entire measured-selection source population.
full=pd.read_sql("""SELECT stay_id,hadm_id,subject_id,age,gender,sofa_24h,heart_rate_max,mbp_min,
resp_rate_max,spo2_min,lactate_max_24h,wbc_max,creatinine_max,invasive_vent,cci,evaluable,hosp_mort,mort_28d
FROM mimiciv_echo.sicm_analysis_dataset""",conn)
conn.close()
full=full.drop(columns=["has_acs"],errors="ignore").merge(all_stays,on=["stay_id","hadm_id"],validate="one_to_one")
full=full[(full.age>=18)&(full.age<=90)&(full.has_acs_corrected==0)].copy()
psf=["age","sofa_24h","heart_rate_max","mbp_min","resp_rate_max","spo2_min",
     "lactate_max_24h","wbc_max","creatinine_max","cci"]
xps=full[psf].copy()
for c in psf: xps[c]=xps[c].fillna(xps[c].median())
xps["invasive_vent"]=full.invasive_vent.fillna(0)
xps["gender_f"]=full.gender.eq("F").astype(int)
scaler=StandardScaler().fit(xps)
pm=LogisticRegression(max_iter=1500).fit(scaler.transform(xps),full.evaluable)
ps=np.clip(pm.predict_proba(scaler.transform(xps))[:,1],0.02,0.98)
prev=full.evaluable.mean();w=np.where(full.evaluable.eq(1),prev/ps,1.0)
we=w[full.evaluable.eq(1)].copy();lo,hi=np.percentile(we,[1,99])
w[full.evaluable.eq(1)]=np.clip(we,lo,hi);full["selection_weight"]=w
full.to_csv(DATA/"round7_selection_source.csv",index=False)
full[["stay_id","evaluable","selection_weight"]].to_csv(DATA/"round7_selection_weights.csv",index=False)
from sklearn.metrics import roc_auc_score
ess=w[full.evaluable.eq(1)].sum()**2/(w[full.evaluable.eq(1)]**2).sum()
pd.DataFrame({"metric":["ps_auc","weight_min","weight_max","ess_evaluable","pretruncation_max"],
 "value":[roc_auc_score(full.evaluable,ps),w[full.evaluable.eq(1)].min(),w[full.evaluable.eq(1)].max(),ess,we.max()]}).to_csv(RESULTS/"round7_ipw_diagnostics.csv",index=False)
co=co[co.has_acs.eq(0)].copy().reset_index(drop=True)
co=co.merge(full[["stay_id","selection_weight"]],on="stay_id",how="left",validate="one_to_one")
aki=pd.read_csv(DATA/"aki_stage_24h.csv")
co=co.merge(aki,on="stay_id",how="left",validate="one_to_one")
co["aki_stage_24h"]=co.aki_stage_24h.fillna(0).astype(int)
co["gender_f"]=co.gender.eq("F").astype(int)
co["lact_f"]=co.lactate_max_24h.fillna(co.lactate_max_24h.median())
co["shock_24h"]=(co.ne_max_24h_strict>0).astype(int)
co.to_csv(DATA/"round7_cohort_timing.csv",index=False)

selection=[]
for value,g in full.groupby("evaluable"):
    selection.append({"evaluable":int(value),"n":len(g),"hospital_deaths":int(g.hosp_mort.sum()),
      "hospital_mortality":g.hosp_mort.mean(),"deaths_28d":int(g.mort_28d.sum()),
      "mortality_28d":g.mort_28d.mean(),"age_median":g.age.median(),"sofa_24h_median":g.sofa_24h.median()})
pd.DataFrame(selection).to_csv(RESULTS/"round7_selection_outcomes.csv",index=False)
checks={"source_stays":len(all_stays),"corrected_acs_flags":int(all_stays.has_acs_corrected.sum()),
        "selection_source_n":len(full),"descriptive_n":len(co),"unique_patients":co.subject_id.nunique(),
        "sicm_n":int(co.sicm.sum()),"current_no_hf_n":int(((co.has_hf==0)&(co.congestive_heart_failure==0)).sum()),
        "current_no_hf_sicm":int(co.loc[(co.has_hf==0)&(co.congestive_heart_failure==0),"sicm"].sum()),
        "postsepsis_ef50_before_sepsis":int((co.ef50_h<co.sepsis_h).sum()),
        "pre_icu_ne_starts_included":0,"pre_icu_vent_starts_included":0,
        "cardiac_surgery_vad_exclusion_implemented":0}
pd.DataFrame({"metric":list(checks),"value":list(checks.values())}).to_csv(RESULTS/"round7_preparation_checks.csv",index=False)
print(pd.Series(checks).to_string())
print("[round7_prepare complete]")
