# -*- coding: utf-8 -*-
"""Time-dependent and 24-h landmark 28-day mortality sensitivity analyses."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "exec" / "data"
RESULTS = ROOT / "exec" / "results"
co = pd.read_csv(DATA / "round7_cohort_timing.csv")
d = co[(co.has_hf.eq(0)) & (co.congestive_heart_failure.eq(0))].copy().reset_index(drop=True)
d["gender_f"] = d.gender.eq("F").astype(int)
checks = {"initial_n":len(d)}

# Do not convert impossible/missing death times into invented event times.
invalid_death = d.death_h.lt(0) | (d.mort_28d.eq(1) & d.death_h.isna())
checks["excluded_invalid_death_time"] = int(invalid_death.sum())
d = d[~invalid_death].copy()
d["death_event"] = d.death_h.notna() & d.death_h.le(672)
d["end_h"] = np.minimum(d.death_h.fillna(672),672)
invalid_onset = d.sicm.eq(1) & (
    d.sicm_onset_h.isna() | d.sicm_onset_h.lt(0) | d.sicm_onset_h.gt(672) |
    (d.death_event & d.sicm_onset_h.ge(d.end_h)))
checks["excluded_invalid_or_postdeath_sicm_onset"] = int(invalid_onset.sum())
d = d[~invalid_onset & d.end_h.gt(0)].copy().reset_index(drop=True)
checks["time_dependent_analysis_n"] = len(d)
checks["time_dependent_sicm_n"] = int(d.sicm.sum())
checks["time_dependent_deaths"] = int(d.death_event.sum())

segments = []
for row in d.itertuples(index=False):
    base = {"stay_id":row.stay_id,"subject_id":row.subject_id,"age":row.age,
            "gender_f":row.gender_f,"cci":row.cci}
    onset = row.sicm_onset_h if row.sicm == 1 else np.nan
    if row.sicm == 1 and onset > 0:
        segments.append({**base,"start":0.0,"stop":float(onset),"sicm_td":0,"event":0})
        start = float(onset)
    else:
        start = 0.0
    segments.append({**base,"start":start,"stop":float(row.end_h),
                     "sicm_td":int(row.sicm == 1),"event":int(row.death_event)})
tv = pd.DataFrame(segments)
# Zero-length segments at simultaneous onset/censoring carry no follow-up information.
tv = tv[tv.stop > tv.start].copy()
# R survival is used for patient-clustered variance in the start-stop model.
tv.to_csv(DATA / "round7_time_varying.csv",index=False)

# Landmark targets early (0-24 h) SICM; later SICM remains in the no-early-SICM group.
lm = d[d.end_h.gt(24)].copy()
lm["early_sicm"] = (lm.sicm.eq(1) & lm.sicm_onset_h.le(24)).astype(int)
lm["duration"] = lm.end_h-24
checks["landmark_n"] = len(lm)
checks["landmark_early_sicm"] = int(lm.early_sicm.sum())
checks["landmark_later_sicm"] = int((lm.sicm.eq(1)&lm.early_sicm.eq(0)).sum())
checks["landmark_deaths"] = int(lm.death_event.sum())
lm.to_csv(DATA / "round7_landmark_mortality.csv",index=False)
pd.DataFrame({"metric":list(checks),"value":list(checks.values())}).to_csv(
    RESULTS / "round7_time_mortality_checks.csv",index=False)
print(pd.Series(checks).to_string())
print("[round7_time_mortality data prepared]")
