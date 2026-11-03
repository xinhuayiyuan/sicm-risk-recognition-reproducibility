# -*- coding: utf-8 -*-
"""Verify pre-ICU and pre-suspected-infection normal LVEF availability."""
import os
from pathlib import Path
import pandas as pd
import psycopg2
ROOT=Path(__file__).resolve().parents[1]
co=pd.read_csv(ROOT/"exec"/"data"/"round7_cohort_timing.csv")
dn=co[(co.has_hf==0)&(co.congestive_heart_failure==0)&(co.sicm==1)]
ids=[int(x) for x in dn.stay_id]
con=psycopg2.connect(dbname="mimic",host="127.0.0.1",port=5431,user="postgres",password=os.environ["MIMIC_PASSWORD"])
q="""SELECT d.stay_id,
MIN(EXTRACT(EPOCH FROM (i.intime-e.measurement_datetime))/86400.0) AS days_before_icu,
MIN(EXTRACT(EPOCH FROM (s.suspected_infection_time-e.measurement_datetime))/86400.0) AS days_before_sepsis
FROM mimiciv_echo.sicm_analysis_dataset d
JOIN mimiciv_echo.lvef e ON e.subject_id=d.subject_id
JOIN mimiciv_icu.icustays i ON i.stay_id=d.stay_id
JOIN mimiciv_echo.sepsis_icu_cohort s ON s.stay_id=d.stay_id
WHERE d.stay_id=ANY(%s) AND e.measurement_datetime<i.intime
AND e.result ~ '^[0-9]+(\\.[0-9]+)?$' AND e.result::numeric>=50
GROUP BY d.stay_id,i.intime,s.suspected_infection_time"""
x=pd.read_sql(q,con,params=(ids,));con.close()
out=pd.DataFrame({"metric":["sicm_n","any_pre_icu_lvef_ge50","pre_icu_within_7d_lvef_ge50"],
                  "value":[len(dn),len(x),int((x.days_before_icu<=7).sum())]})
out.to_csv(ROOT/"exec"/"results"/"round7_baseline_echo.csv",index=False)
print(out.to_string(index=False))
