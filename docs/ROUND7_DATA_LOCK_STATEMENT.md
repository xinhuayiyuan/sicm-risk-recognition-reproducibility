# Round-7 derived-data lock statement

Locked at: 2026-11-03T09:44:52

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

| path                                    |   bytes | sha256                                                           |
|:----------------------------------------|--------:|:-----------------------------------------------------------------|
| exec\data\round7_cohort_timing.csv      |  720480 | 66EAE0F4E0A45597FB67D55E133FC0FDA66C0BE65841BA6240A6A97BAD978E09 |
| exec\data\round7_selection_source.csv   | 1498320 | 419CA546BD84012AB33127BB03CA9590EED4E85E3EA01D3EB8F273D51386D0DB |
| exec\data\round7_comprisk.csv           |   35665 | 9BB2006F401F2FF500E0634793F2B345571B559E43CA77CE8EE4B7DCA9870028 |
| exec\data\round7_mi_dataset.csv         |   26992 | 7192AD8478A7B67BC44F546E112D41AA2BBD3D2B6D796EAEBEC7683BD30908CC |
| exec\data\round7_comprisk_status.csv    |  333475 | C2AC507A86CAFE2A2CA61B1BA5F910E3EE56FB87EDE004281F48EAF88089B565 |
| exec\data\round7_time_varying.csv       |   44750 | 13CA8AF8AC91D220C065FE2F5A374D88875F8B1157D7F0DBDB08A6C74EC674B1 |
| exec\data\round7_landmark_mortality.csv |  338247 | F2EE289FA400364A00B49C39BDE5132A71141895172060BA224295F45BBF0181 |

## Integrity and post-lock rules

The listed files were set to read-only after verification. Rebuilding requires a new analysis version and new hashes; it must not silently overwrite the locked version. The original seed is 20260101. Aggregate outputs, code, figures and manuscript files are covered by the existing checksum manifests.

## Remaining limitations

The exact baseline/up-titration algorithm of legacy `vaso_new_upgrade` and the complete eICU export construction remain unavailable. These are disclosed reconstruction limitations, not silently repaired components. Round-7 is a post-hoc corrected analysis and is not retrospectively prespecified.
