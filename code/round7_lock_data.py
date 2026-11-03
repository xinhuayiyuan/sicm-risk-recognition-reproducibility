# -*- coding: utf-8 -*-
"""Lock the verified Round-7 derived data and create an auditable lock statement."""
from pathlib import Path
from datetime import datetime
import hashlib
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"exec"/"data";RESULTS=ROOT/"exec"/"results";PKG=ROOT/"投稿包_BMC"
input_names=[
 "round7_cohort_timing.csv","round7_selection_source.csv","round7_comprisk.csv",
 "round7_mi_dataset.csv","round7_comprisk_status.csv","round7_time_varying.csv",
 "round7_landmark_mortality.csv",
]
input_paths=[DATA/name for name in input_names]
for p in input_paths:
    assert p.exists(),p
co=pd.read_csv(DATA/"round7_cohort_timing.csv")
dn=co[(co.has_hf==0)&(co.congestive_heart_failure==0)]
assert (len(co),co.subject_id.nunique(),int(co.sicm.sum()))==(1297,1271,466)
assert (len(dn),dn.subject_id.nunique(),int(dn.sicm.sum()))==(613,606,111)
assert int((co.ef50_h<co.sepsis_h).sum())==0
checks=pd.read_csv(RESULTS/"round7_preparation_checks.csv").set_index("metric").value
assert checks.corrected_acs_flags==2772
assert checks.pre_icu_ne_starts_included==0 and checks.pre_icu_vent_starts_included==0

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest().upper()
rows=[]
for p in input_paths:
    rows.append({"path":str(p.relative_to(ROOT)),"bytes":p.stat().st_size,"sha256":sha(p)})
manifest=pd.DataFrame(rows)
manifest.to_csv(RESULTS/"round7_lock_manifest.csv",index=False)

statement=f"""# Round-7 derived-data lock statement

Locked at: {datetime.now().isoformat(timespec='seconds')}

## Scope

This lock covers the verified derived Round-7 analysis inputs, not the source MIMIC/eICU databases and not the legacy `vaso_new_upgrade` or eICU `temp_final_export` construction. No record-level data are added to the public package.

## Locked analysis definitions

- ACS: ICD-9 410/411 and ICD-10 I21/I22; 2,772 flagged stays in the source dataset.
- Descriptive cohort: 1,297 stays in 1,271 patients, with 466 operational SICM events.
- Current no-index-admission-HF cohort: 613 stays in 606 patients, with 111 SICM events.
- Qualifying low LVEF and decline follow-up are at/after suspected infection; audit count with EF<50 before suspected infection is zero.
- NE and ventilation starts included before ICU: zero. Treatment starts are restricted to ICU 0–24 h.
- Selection source population: 14,980 stays, with 1,297 evaluable.
- No separate cardiac-surgery/transplant/VAD exclusion was implemented.

## Locked input files and hashes

{manifest.to_markdown(index=False)}

## Integrity and post-lock rules

The listed files were set to read-only after verification. Rebuilding requires a new analysis version and new hashes; it must not silently overwrite the locked version. The original seed is 20260101. Aggregate outputs, code, figures and manuscript files are covered by the existing checksum manifests.

## Remaining limitations

The exact baseline/up-titration algorithm of legacy `vaso_new_upgrade` and the complete eICU export construction remain unavailable. These are disclosed reconstruction limitations, not silently repaired components. Round-7 is a post-hoc corrected analysis and is not retrospectively prespecified.
"""
for path in [ROOT/"exec"/"ROUND7_DATA_LOCK_STATEMENT.md",ROOT/"github_upload"/"docs"/"ROUND7_DATA_LOCK_STATEMENT.md",ROOT/"reproducibility_package"/"docs"/"ROUND7_DATA_LOCK_STATEMENT.md"]:
    path.write_text(statement,encoding="utf-8")
for p in input_paths:
    p.chmod(p.stat().st_mode & ~0o222)
print("Locked derived inputs:",len(input_paths))
print(manifest.to_string(index=False))
