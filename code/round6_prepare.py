# -*- coding: utf-8 -*-
"""Prepare HF/event timing. Requires MIMIC_PASSWORD; record-level output stays local."""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import psycopg2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "exec" / "data"
RESULTS = ROOT / "exec" / "results"
co = pd.read_csv(DATA / "cohort_main.csv")
subjects = [int(x) for x in co.subject_id.unique()]
stays = [int(x) for x in co.stay_id.unique()]
conn = psycopg2.connect(dbname="mimic", host="127.0.0.1", port=5431,
                        user="postgres", password=os.environ["MIMIC_PASSWORD"], connect_timeout=10)
conn.cursor().execute("SET statement_timeout = '120s'")
admissions = pd.read_sql(
    "SELECT subject_id, hadm_id, admittime FROM mimiciv_hosp.admissions WHERE subject_id = ANY(%s)",
    conn, params=(subjects,))
hf_condition = """(
 (z.icd_version=9 AND (z.icd_code='39891' OR
  z.icd_code IN ('40201','40211','40291','40401','40403','40411','40413','40491','40493') OR
  z.icd_code IN ('4254','4255','4256','4257','4258','4259') OR z.icd_code LIKE '428%%')) OR
 (z.icd_version=10 AND (z.icd_code IN
  ('I099','I110','I130','I132','I255','I420','I425','I426','I427','I428','I429','I43')
  OR z.icd_code LIKE 'I50%%')))"""
hf_admissions = pd.read_sql(
    "SELECT DISTINCT a.subject_id,z.hadm_id FROM mimiciv_hosp.diagnoses_icd z "
    "JOIN mimiciv_hosp.admissions a ON z.hadm_id=a.hadm_id WHERE a.subject_id=ANY(%s) AND " + hf_condition,
    conn, params=(subjects,))
admissions["admittime"] = pd.to_datetime(admissions.admittime)
hf_hadm = set(hf_admissions.hadm_id.astype(int))
prior_hf_hadm = set()
for row in co[["subject_id", "hadm_id"]].itertuples(index=False):
    current = admissions.loc[admissions.hadm_id.eq(row.hadm_id), "admittime"].iloc[0]
    earlier = admissions[(admissions.subject_id == row.subject_id) & (admissions.admittime < current)]
    if any(int(h) in hf_hadm for h in earlier.hadm_id):
        prior_hf_hadm.add(int(row.hadm_id))
co["index_hf"] = co.hadm_id.astype(int).isin(hf_hadm).astype(int)
co["prior_adm_hf"] = co.hadm_id.astype(int).isin(prior_hf_hadm).astype(int)
co["hf_time_group"] = np.select(
    [co.prior_adm_hf.eq(1), co.index_hf.eq(1)],
    ["documented_prior_hf", "index_first_hf_time_uncertain"], default="no_hf_diagnosis_record")

timing = pd.read_sql("""
SELECT i.stay_id,
 EXTRACT(EPOCH FROM (s.suspected_infection_time-i.intime))/3600.0 AS sepsis_h,
 EXTRACT(EPOCH FROM (a.deathtime-i.intime))/3600.0 AS deathtime_h,
 EXTRACT(EPOCH FROM (p.dod-i.intime))/3600.0 AS dod_h
FROM mimiciv_icu.icustays i
JOIN mimiciv_hosp.admissions a ON i.hadm_id=a.hadm_id
JOIN mimiciv_hosp.patients p ON i.subject_id=p.subject_id
JOIN mimiciv_echo.sepsis_icu_cohort s ON i.stay_id=s.stay_id WHERE i.stay_id=ANY(%s)
""", conn, params=(stays,))
ef50 = pd.read_sql("""
SELECT stay_id,MIN(hours_from_icu_intime) AS ef50_h FROM mimiciv_echo.echo_lvef_numeric
WHERE stay_id=ANY(%s) AND lvef_value<50 AND hours_from_icu_intime>=0 GROUP BY stay_id
""", conn, params=(stays,))
decline = pd.read_sql("""
SELECT stay_id,MIN(followup_hours) AS decline_h FROM mimiciv_echo.echo_lvef_followup
WHERE stay_id=ANY(%s) AND lvef_drop_absolute>=10 AND followup_hours>=0 GROUP BY stay_id
""", conn, params=(stays,))
trop = pd.read_sql("""
SELECT i.stay_id,MIN(EXTRACT(EPOCH FROM (t.charttime-i.intime))/3600.0) AS trop_positive_h
FROM mimiciv_echo.troponin_sepsis t JOIN mimiciv_icu.icustays i ON t.stay_id=i.stay_id
WHERE t.stay_id=ANY(%s) AND t.valuenum>0.014 GROUP BY i.stay_id
""", conn, params=(stays,))
for frame in [timing, ef50, decline, trop]:
    co = co.merge(frame, on="stay_id", how="left", validate="one_to_one")
co["imaging_h"] = co.ef50_h
co.loc[co.sicm.eq(1) & co.imaging_h.isna(), "imaging_h"] = co.decline_h
co["death_h"] = co.deathtime_h.fillna(co.dod_h)
co["sicm_onset_h"] = np.nan
mask = co.sicm.eq(1)
co.loc[mask, "sicm_onset_h"] = co.loc[mask, ["sepsis_h", "imaging_h", "trop_positive_h"]].max(axis=1)


# Reproduce the locked measured-selection weights without outcome-driven refitting.
full = pd.read_sql("""
SELECT stay_id,age,gender,sofa_24h,heart_rate_max,mbp_min,resp_rate_max,spo2_min,
 lactate_max_24h,wbc_max,creatinine_max,invasive_vent,cci,evaluable,hosp_mort,mort_28d
FROM mimiciv_echo.sicm_analysis_dataset WHERE age>=18 AND age<=90 AND has_acs=0
""", conn)
conn.close()
psf = ["age","sofa_24h","heart_rate_max","mbp_min","resp_rate_max","spo2_min",
       "lactate_max_24h","wbc_max","creatinine_max","cci"]
xps = full[psf].copy()
for c in psf:
    xps[c] = xps[c].fillna(xps[c].median())
xps["invasive_vent"] = full.invasive_vent.fillna(0)
xps["gender_f"] = full.gender.eq("F").astype(int)
scaler = StandardScaler().fit(xps)
ps_model = LogisticRegression(max_iter=1500).fit(scaler.transform(xps), full.evaluable)
ps = np.clip(ps_model.predict_proba(scaler.transform(xps))[:, 1], 0.02, 0.98)
prevalence = full.evaluable.mean()
weights = np.where(full.evaluable.eq(1), prevalence / ps, 1.0)
eval_weights = weights[full.evaluable.eq(1)].copy()
lo, hi = np.percentile(eval_weights, [1, 99])
weights[full.evaluable.eq(1)] = np.clip(eval_weights, lo, hi)
full["selection_weight"] = weights
co = co.merge(full[["stay_id","selection_weight"]], on="stay_id", how="left", validate="one_to_one")
selection_rows = []
for value, group in full.groupby("evaluable"):
    selection_rows.append({"evaluable":int(value),"n":len(group),
        "hospital_deaths":int(group.hosp_mort.sum()),"hospital_mortality":group.hosp_mort.mean(),
        "deaths_28d":int(group.mort_28d.sum()),"mortality_28d":group.mort_28d.mean(),
        "age_median":group.age.median(),"sofa_24h_median":group.sofa_24h.median()})
pd.DataFrame(selection_rows).to_csv(RESULTS / "round6_selection_outcomes.csv",index=False)
aki = pd.read_csv(DATA / "aki_stage_24h.csv")
st = pd.read_csv(DATA / "cohort_dn_strict24h.csv", usecols=["stay_id","ne_max_24h_strict","vent_24h"])
co = co.merge(aki, on="stay_id", how="left", validate="one_to_one")
co = co.merge(st, on="stay_id", how="left", validate="one_to_one")
co["aki_stage_24h"] = co.aki_stage_24h.fillna(0).astype(int)
co["vent_24h"] = co.vent_24h.fillna(0).astype(int)
co["gender_f"] = co.gender.eq("F").astype(int)
co["lact_f"] = co.lactate_max_24h.fillna(co.lactate_max_24h.median())
co.to_csv(DATA / "round6_cohort_timing.csv", index=False)
summary = {
    "n": len(co),
    "documented_prior_hf": int(co.prior_adm_hf.sum()),
    "index_first_hf_time_uncertain": int((co.prior_adm_hf.eq(0) & co.index_hf.eq(1)).sum()),
    "no_hf_diagnosis_record": int((co.prior_adm_hf.eq(0) & co.index_hf.eq(0)).sum()),
    "current_no_hf_but_prior_admission_hf": int(((co.has_hf.eq(0) & co.congestive_heart_failure.eq(0)) & co.prior_adm_hf.eq(1)).sum()),
    "sicm_with_complete_onset": int((co.sicm.eq(1) & co.sicm_onset_h.notna()).sum()),
    "negative_death_times": int(co.death_h.lt(0).sum()),
}
pd.DataFrame({"metric":list(summary),"value":list(summary.values())}).to_csv(
    RESULTS / "round6_preparation_checks.csv", index=False)
print(pd.Series(summary).to_string())
print(co.hf_time_group.value_counts().to_string())
print("[round6_prepare complete]")
