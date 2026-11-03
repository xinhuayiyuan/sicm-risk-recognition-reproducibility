# -*- coding: utf-8 -*-
"""Render round-6 aggregate outputs without manual transcription."""
import json
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "exec" / "results"
a = pd.read_csv(R/"round6_hf_timing_association.csv").set_index("scheme")
p = pd.read_csv(R/"round6_hf_timing_prediction.csv").set_index("analysis")
v = pd.read_csv(R/"round6_patient_level_validation.csv").set_index("analysis")
t = pd.read_csv(R/"round6_time_mortality.csv").set_index("analysis")
s = pd.read_csv(R/"round6_selection_outcomes.csv").set_index("evaluable")
co = pd.read_csv(ROOT/"exec"/"data"/"round6_cohort_timing.csv")
def ci(r,stem,digits=2): return f"{r[stem]:.{digits}f} ({r[stem+'_lo']:.{digits}f}–{r[stem+'_hi']:.{digits}f})"
def auc(r): return f"{r.AUC:.3f} ({r.AUC_lo:.3f}–{r.AUC_hi:.3f})"
labels={"A_current_index_admission_exclusion":"A: current index-admission exclusion",
"B_strict_time_exclude_uncertain":"B: strict-time; uncertain HF excluded",
"C_expanded_possible_new_onset":"C: expanded possible new-onset"}
lines=["## Supplement S9. Post-hoc HF-timing, patient-level and exposure-timing sensitivity analyses","",
"These post-hoc analyses were not used to select a scheme by significance or AUC. Earlier admissions were identified within MIMIC using the Quan/Charlson HF code set across ICD eras. Absence of an earlier HF code does not establish absence of clinical HF because MIMIC lacks complete outpatient and pre-database history.","","### S9a. HF time classification","",
"| HF record classification | ICU stays | SICM events |","|---|---:|---:|"]
for key,label in [("documented_prior_hf","Documented HF in an earlier admission"),("index_first_hf_time_uncertain","HF first coded during index admission; timing uncertain"),("no_hf_diagnosis_record","No HF diagnosis record in available admissions")]:
 g=co[co.hf_time_group.eq(key)];lines.append(f"| {label} | {len(g)} | {int(g.sicm.sum())} |")
lines+=["","Scheme A reproduces current index-admission exclusion. Scheme B excludes earlier-admission HF and index-first/time-uncertain HF. Scheme C excludes earlier-admission HF but includes index-first HF as possible new-onset disease.","","### S9b. HF classification sensitivity","",
"Association models used the original adjustment set and patient-clustered standard errors. IPW used locked measured-selection weights. Risk differences used g-computation with 500 patient-cluster bootstrap replicates. Prediction used five-fold patient-grouped outer CV and ten-fold patient-grouped inner penalty selection; AUC intervals used 2,000 patient-cluster bootstrap replicates.","",
"| Scheme | n / patients | SICM | Deaths, SICM / reference | Adjusted OR (95% CI) | IPW OR (95% CI) | Adjusted RD, pp (95% CI) | OOF AUC (95% CI) | Cal. slope | ICI |","|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
for key in labels:
 x,y=a.loc[key],p.loc[key]
 lines.append(f"| {labels[key]} | {int(x.n)} / {int(x.patients)} | {int(x.sicm_events)} | {int(x.deaths_sicm)} / {int(x.deaths_reference)} | {ci(x,'adjusted_OR')} | {ci(x,'IPW_OR')} | {100*x.adjusted_RD:.1f} ({100*x.adjusted_RD_lo:.1f} to {100*x.adjusted_RD_hi:.1f}) | {auc(y)} | {y.calibration_slope:.3f} | {y.ICI:.3f} |")
lines+=["","These schemes address record-timing classification only; none proves clinical de novo dysfunction.","","### S9c. Patient-level internal validation","",
"Patient-grouped partitioning kept all stays from one patient in the same development/test or inner-CV partition. Confidence intervals resampled patients rather than stays.","",
"| Evaluation | n / patients | Events | AUC (95% CI) | PR-AUC | Brier | Cal. intercept | Cal. slope | ICI |","|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
for key,label in [("patient_grouped_holdout","Patient-grouped held-out test"),("patient_grouped_oof","Patient-grouped five-fold OOF")]:
 x=v.loc[key];lines.append(f"| {label} | {int(x.n)} / {int(x.patients)} | {int(x.events)} | {auc(x)} | {x.PR_AUC:.3f} | {x.Brier:.3f} | {x.calibration_intercept:.3f} | {x.calibration_slope:.3f} | {x.ICI:.3f} |")
lines+=["","For reference, original stay-level held-out and OOF AUCs were 0.731 (0.655–0.795) and 0.677 (0.635–0.719). Different splits are not independent head-to-head comparisons.","","### S9d. Mortality exposure-timing sensitivity","",
"SICM onset was the latest of suspected-infection time, qualifying imaging time and first troponin above threshold. Two records with impossible negative death times and one SICM record without valid pre-death onset were excluded. Cox models adjusted only for age, sex and Charlson index, determinable before exposure, and used patient-clustered robust variance. The landmark analysis targeted SICM by 24 h; later-onset SICM remained in the no-early-SICM group.","",
"| Analysis | n / patients | Exposure events | Deaths | HR (95% CI) | P |","|---|---:|---:|---:|---:|---:|"]
for key,label in [("time_dependent_cox","Time-dependent SICM, ICU-admission origin"),("landmark_24h_early_sicm","24-h landmark, early SICM")]:
 x=t.loc[key];lines.append(f"| {label} | {int(x.n)} / {int(x.patients)} | {int(x.sicm_events)} | {int(x.deaths)} | {ci(x,'HR')} | {x.p:.4g} |")
lines+=["","These HRs use a reduced baseline adjustment set and are not numerically equivalent to the primary fixed-exposure OR.","","### S9e. Evaluability and early outcomes","",
"| Evaluable for troponin and LVEF | n | Hospital deaths (%) | 28-day deaths (%) | Median age | Median SOFA-24h |","|---|---:|---:|---:|---:|---:|"]
for key,label in [(0,"No"),(1,"Yes")]:
 x=s.loc[key];lines.append(f"| {label} | {int(x.n)} | {int(x.hospital_deaths)} ({100*x.hospital_mortality:.1f}) | {int(x.deaths_28d)} ({100*x.mortality_28d:.1f}) | {x.age_median:.1f} | {x.sofa_24h_median:.1f} |")
lines+=["","These are descriptive differences, not estimates of the causal effect of echocardiography. IPW remains limited to measured selection factors.","","### S9f. Mediator timing","",
"The stored MIMIC mediators are aggregated after sepsis onset rather than after first complete SICM onset. A valid exposure-to-mediator sequence therefore cannot be established for all records. No new causal natural-effect analysis was performed; the existing decomposition is exploratory and associational, not proof of mediation or its absence.",""]
(R/"round6_supplement_snippet.md").write_text("\n".join(lines),encoding="utf-8")
main={"hf":"; ".join(f"{labels[k].split(':')[0]} {ci(a.loc[k],'adjusted_OR')}" for k in labels),
"patient":f"patient-grouped held-out AUC {auc(v.loc['patient_grouped_holdout'])} and OOF AUC {auc(v.loc['patient_grouped_oof'])}",
"time":f"time-dependent Cox HR {ci(t.loc['time_dependent_cox'],'HR')}; 24-h landmark HR {ci(t.loc['landmark_24h_early_sicm'],'HR')}"}
(R/"round6_main_snippets.json").write_text(json.dumps(main,indent=2),encoding="utf-8")
print("\n".join(lines));print(json.dumps(main,indent=2))
