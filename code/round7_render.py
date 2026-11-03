# -*- coding: utf-8 -*-
"""Render Round-7 manuscript, supplement and tables exclusively from verified outputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
PKG=ROOT/"投稿包_BMC";R=ROOT/"exec"/"results";D=ROOT/"exec"/"data"
co=pd.read_csv(D/"round7_cohort_timing.csv")
dn=co[(co.has_hf==0)&(co.congestive_heart_failure==0)].copy()
def load(name): return pd.read_csv(R/name)
model=load("round7_model_fulleval.csv")
metrics=load("round7_analysis_set_metrics.csv").set_index("analysis_set")
valid=load("round7_patient_level_validation.csv").set_index("analysis")
hfa=load("round7_hf_timing_association.csv").set_index("scheme")
hfp=load("round7_hf_timing_prediction.csv").set_index("analysis")
time=load("round7_time_mortality.csv").set_index("analysis")
adj=load("round7_adj_or.csv").set_index("item").value
iptw=load("round7_iptw.csv")
med=load("round7_mediation_boot2000.csv").set_index("mediator")
selection=load("round7_selection_outcomes.csv").set_index("evaluable")
score_map=load("round7_score_map.csv").set_index("item").value
artifact=json.loads((R/"round7_primary_lasso_model.json").read_text(encoding="utf-8"))

def ci(r,stem): return f"{r[stem]:.2f} ({r[stem+'_lo']:.2f}–{r[stem+'_hi']:.2f})"
def auc3(r): return f"{r.AUC:.3f} ({r.AUC_lo:.3f}–{r.AUC_hi:.3f})"
def ci_ipw(r): return f"{r.OR_iptw:.2f} ({r.lo:.2f}–{r.hi:.2f})"
def pct(a,b): return f"{100*a/b:.1f}%"
def md_table(headers,rows):
    return "\n".join(["| "+" | ".join(headers)+" |","|"+"|".join(["---"]*len(headers))+"|"]+["| "+" | ".join(map(str,row))+" |" for row in rows])
def model_row(name):
    r=model[model.model.eq(name)].iloc[0]
    return [name,f"{r.AUC:.3f} ({r.AUC_lo:.3f}–{r.AUC_hi:.3f})",f"{r.PR_AUC:.3f}",f"{r.Brier:.3f}",f"{r.Cal_intercept:.3f}",f"{r.Cal_slope:.3f}",f"{r.ICI:.3f}"]

# Verified cohort and analysis-set values.
n_all=len(co);n_patients=co.subject_id.nunique();n_sicm=int(co.sicm.sum())
n_dn=len(dn);n_dn_patients=dn.subject_id.nunique();n_events=int(dn.sicm.sum());n_ref=int((dn.sicm==0).sum())
n_hf=n_all-n_dn;n_hf_sicm=n_sicm-n_events
deaths_s=int(dn.loc[dn.sicm.eq(1),"mort_28d"].sum());deaths_r=int(dn.loc[dn.sicm.eq(0),"mort_28d"].sum())
legacy=co[co.has_hf==0]
strict_n=int(hfa.loc["B_strict_time_exclude_uncertain","n"])
expanded_n=int(hfa.loc["C_expanded_possible_new_onset","n"])
time_groups=co.groupby("hf_time_group").agg(n=("sicm","size"),events=("sicm","sum"))
threshold=metrics.loc["Test threshold nearest sensitivity 0.65 (exploratory)"]
main_auc=model_row("LASSO-logistic (lambda.min; primary)")
oof_auc=auc3(metrics.loc["Five-fold OOF: primary specification"])
group_test_auc=auc3(valid.loc["patient_grouped_holdout"])
group_oof_auc=auc3(valid.loc["patient_grouped_oof"])

abstract=f"""## Abstract

Background We re-evaluated an operational sepsis-induced cardiomyopathy (SICM) phenotype after a data-chain audit identified incomplete acute coronary syndrome (ACS) exclusion and inconsistent imaging and treatment timestamps.

Methods This post-hoc corrected analysis used Sepsis-3 ICU stays in MIMIC-IV. ACS was excluded using ICD-9 410/411 and ICD-10 I21/I22. Qualifying low LVEF and decline follow-up were required at/after suspected infection. Treatment starts were restricted to ICU 0–24 h. The current no-index-admission-HF cohort contained {n_dn} stays and {n_events} SICM events. A 44-candidate LASSO-logistic model was evaluated by a stay-level split, out-of-fold (OOF) validation and patient-grouped validation. Mortality was analysed with covariate adjustment and measured-selection IPW.

Results SICM mortality was {pct(deaths_s,n_events)} ({deaths_s}/{n_events}) versus {pct(deaths_r,n_ref)} ({deaths_r}/{n_ref}). Test AUC was {main_auc[0]}; OOF AUC was {oof_auc}; patient-grouped OOF AUC was {group_oof_auc}. The adjusted OR was {ci(hfa.loc['A_current_index_admission_exclusion'],'adjusted_OR')}, and IPW OR was {ci(hfa.loc['A_current_index_admission_exclusion'],'IPW_OR')}. Strict-time HF exclusion gave OR {ci(hfa.loc['B_strict_time_exclude_uncertain'],'adjusted_OR')}; expanded possible-new-onset classification attenuated this to {ci(hfa.loc['C_expanded_possible_new_onset'],'adjusted_OR')}. Time-dependent Cox HR was {ci(time.loc['time_dependent_cox'],'HR')}.

Conclusions Corrected data supported an association between operational SICM and 28-day mortality, but discrimination was limited and calibration unstable. HF classification materially affected the association. The model, score and threshold are exploratory and not suitable for clinical triage. Multicentre imaging validation is required.

"""
main_body=f"""# Left Ventricular Systolic Dysfunction in Sepsis without Index-Admission Heart Failure Diagnosis: 24-h Risk Recognition and 28-day Mortality: a retrospective development and internal validation study

Running title: Corrected risk recognition of septic left ventricular dysfunction

Article type: Original Research

Authors: [Author names, degrees and ORCID iDs — to be completed at submission]

Affiliations: [Institutional affiliations to be completed at submission]

Corresponding author: [Name, postal address, telephone, e-mail and ORCID to be completed at submission]

Reporting: TRIPOD+AI-, STROBE- and AGReMA-informed reporting. Round-7 is a post-hoc corrected analysis and is not represented as the original prespecified analysis.

""" + abstract + """## 1. Introduction

Sepsis-induced cardiomyopathy (SICM) definitions vary in HF ascertainment, imaging timing and troponin use. Our internal review found that numerical results could be reproduced while upstream definitions remained inconsistent. Round-7 therefore corrects ACS ascertainment across coding eras, restricts qualifying imaging to the post-suspected-infection period and applies explicit ICU time bounds to treatment variables. The objective is to reassess 0–24-h risk recognition and 28-day mortality rather than preserve earlier estimates.

## 2. Methods

### 2.1 Data and cohort corrections

MIMIC-IV v3.1 and MIMIC-IV-Echo were used. Of 18,273 Sepsis-3 ICU stays, 2,350 had troponin and structured LVEF records. ACS was excluded using ICD-9 410/411 and ICD-10 I21/I22. The corrected descriptive cohort comprised 1,297 stays in 1,271 patients. The current no-index-admission-HF cohort comprised 613 stays in 606 patients (111 SICM and 502 references); the index-admission HF group contained 684 stays (355 SICM). No separate cardiac-surgery/transplant/VAD exclusion was implemented because no auditable code list was available; no such exclusion is claimed.

Earlier-admission HF was separated from index-first HF using the Quan/Charlson code set. In the corrected descriptive cohort, 280 stays had earlier-admission HF, 436 had index-first/time-uncertain HF and 581 had no available HF diagnosis record. These are hospital-record classifications, not complete clinical histories.

### 2.2 Phenotype and time windows

Operational SICM required troponin T >0.014 ng/mL and either post-suspected-infection LVEF <50% or a ≥10-point decline from the first ICU echocardiogram, with qualifying follow-up also at/after suspected infection. This produced 466 events in the descriptive cohort and 111 in the current no-index-HF cohort. Nine cases qualified by decline alone. Only 7 cases had LVEF ≥50% within 7 days before ICU; 26 had any earlier pre-ICU LVEF ≥50%. Thus, de novo dysfunction was not independently confirmed for most cases.

Treatment starts were required within ICU 0–24 h. AKI stage used KDIGO measurements within that window. Continuous predictors were imputed and standardised using development-set statistics. Forty-four candidates were evaluated; dobutamine and milrinone were not used in the locked model.

### 2.3 Model development and validation

The original stratified split contained 429 development stays/78 events and 184 test stays/33 events. LASSO-logistic regression used ten-fold penalty selection. Five-fold OOF validation used all 613 stays. Additional patient-grouped holdout and nested five-fold OOF analyses kept all stays from each patient in the same partition. AUC, PR-AUC, Brier score, calibration intercept/slope and ICI were reported. The integer score used pure coefficient rounding; its forced-sign variant is retained only in the results archive. The operating threshold was selected in the test set to target sensitivity closest to 0.65 and is exploratory.

### 2.4 Mortality, selection weighting and HF sensitivity

Adjusted logistic models used age, SOFA-24h, Charlson index, lactate, ventilation and AKI stage. IPW used the corrected 14,980-stay selection source population, with 1st/99th-percentile truncation and patient-clustered robust variance. A current-specification, patient-clustered full-procedure bootstrap repeated propensity fitting 300 times. Risk differences used 500 patient-cluster bootstrap replicates. HF classification was compared under current (A), strict-time (B) and expanded possible-new-onset (C) rules without selecting a scheme by significance or AUC.

### 2.5 Exposure timing and pathway analyses

SICM onset was the latest of suspected infection, qualifying imaging and first positive troponin. Time-dependent and 24-h-landmark Cox models used age, sex and Charlson index with patient-clustered robust variance. Exploratory g-computation decomposed vasopressor pathways but was not interpreted as identified causal mediation because exposure–mediator order was not guaranteed. eICU remained a proxy-pathway analysis, not external validation of the imaging phenotype or model.

"""
main_body += """## 3. Results

### 3.1 Cohort and mortality

In the current cohort, 111/613 stays (18.1%) had operational SICM. Deaths were 53/111 (47.7%) versus 180/502 (35.9%). The adjusted OR was 1.78 (1.15–2.76), patient-clustered IPW OR 1.94 (1.14–3.29), and adjusted risk difference 12.7 percentage points (2.3 to 22.3). Strict-time HF exclusion produced a similar estimate (OR 1.77, 1.12–2.79), whereas expanded possible-new-onset classification attenuated it (OR 1.09, 0.82–1.45). Time-dependent Cox HR was 1.96 (1.42–2.70), and 24-h landmark HR was 2.13 (1.45–3.15); these HRs used a reduced baseline adjustment set.

### 3.2 Risk recognition and calibration

The primary model test AUC was 0.634 (0.530–0.738), PR-AUC 0.314 and Brier score 0.153. Stay-level OOF AUC was 0.616 (0.556–0.672). Patient-grouped holdout AUC was 0.539 (0.435–0.639) and patient-grouped OOF AUC 0.590 (0.530–0.649). All 44 candidates had non-zero penalised coefficients. The global-shrinkage criterion was not met (estimated required n≈2,825). XGBoost and LightGBM also showed marked calibration instability and are not presented as validated alternatives.

Pure-rounding integer score AUC was 0.618 (0.510–0.722). The post-hoc threshold was 0.157735, with sensitivity 63.6%, specificity 61.6% and 79/184 flagged. It was selected and assessed in the same test set and cannot support triage.

### 3.3 Pathway and sensitivity findings

Exploratory MIMIC decomposition showed no positive indirect association through vasopressor initiation/up-titration (NIE −0.18 pp, −1.52 to 1.01) or peak norepinephrine (NIE +0.34 pp, −0.50 to 1.33). These are associational decompositions, not proof of direct biology or absent mediation. eICU retained its previously documented proxy limitations and was not reclassified as external validation.

## 4. Discussion

The corrected analysis illustrates why data-chain validity must precede model interpretation. ACS and imaging-time corrections materially changed the cohort and model performance. The mortality association persisted under strict HF timing and exposure-time sensitivity analyses, but was attenuated by the expanded HF classification. Risk recognition was limited and calibration unstable. Accordingly, the principal conclusion concerns an association within an operational selected cohort, not a clinically deployable prediction rule.

Limitations include the single-centre selected cohort, lack of protocolised pre-sepsis imaging, administrative HF misclassification, residual post-exposure adjustment, unmeasured imaging selection, assay-era drift, small event count, all-44-variable retention and absence of external imaging validation. Cardiac-surgery/transplant/VAD exclusion was not separately implemented. Exploratory mediator decomposition remains limited by temporal ordering and untestable causal assumptions.

## 5. Conclusions

Corrected operational SICM was associated with higher 28-day mortality, but discrimination was limited and estimates varied with HF classification. The current model, integer score and test-set threshold are exploratory and unsuitable for clinical triage. Prospective multicentre cohorts with protocolised early imaging and genuine pre-sepsis LVEF assessment are required before model revision or clinical use.

"""
# Main Tables 1-4, generated from current aggregate outputs.
t1=load("round7_table1_p.csv")
extra=load("round7_table1_extra.csv")
label_map={'n':'n','年龄, 岁':'Age, years','SOFA-24h':'SOFA score, first 24 h','Charlson指数':'Charlson comorbidity index',
'心率max, /min':'Heart rate max, /min','MBPmin, mmHg':'MAP min, mmHg','乳酸max, mmol/L':'Lactate max, mmol/L',
'肌酐max, mg/dL':'Creatinine max, mg/dL','血小板min, 10^9/L':'Platelets min, ×10⁹/L','血红蛋白min, g/dL':'Haemoglobin min, g/dL',
'最低LVEF, %':'Lowest post-infection LVEF, %','ICU住院, d':'ICU stay, days','女性':'Female sex','心肌梗死史':'Prior myocardial infarction',
'慢阻肺':'Chronic pulmonary disease','糖尿病':'Diabetes','恶性肿瘤':'Malignancy','慢性肾病':'Chronic kidney disease',
'有创通气':'Invasive ventilation (whole stay)','多巴酚丁胺':'Dobutamine use (whole stay)',
'血管活性药升级':'Vasopressor initiation/up-titration (post-onset)','住院死亡':'Hospital mortality','28天死亡':'28-day mortality'}
t1rows=[]
for _,r in t1.iterrows():
    t1rows.append([label_map.get(r['变量'],r['变量']),r.iloc[1],r.iloc[2],r.iloc[3],r.P])
t1rows.insert(1,['Weight, kg',extra.iloc[1].g1,extra.iloc[1].g0,extra.iloc[1].g2,extra.iloc[1].p])
t1rows.insert(11,['Peak troponin T, ng/mL',extra.iloc[0].g1,extra.iloc[0].g0,extra.iloc[0].g2,extra.iloc[0].p])
table1="### Table 1. Clinical characteristics in the corrected descriptive cohort (n = 1,297)\n\n"+md_table(
 ["Characteristic","New-onset SICM (n = 111)","Reference, no index-admission HF (n = 502)","Index-admission HF (n = 684; 355 SICM)","P value"],t1rows)+"\n\n"
table2rows=[model_row("Baseline (age+SOFA)"),model_row("LASSO-logistic (lambda.min; primary)"),model_row("LASSO-logistic (1-SE, sparse)"),
            model_row("XGBoost (Platt-calibrated)"),model_row("LightGBM (Platt-calibrated)")]
table2="### Table 2. Model performance in the current no-index-admission-HF cohort (n = 613; 111 events)\n\n"+md_table(
 ["Model / evaluation","AUC (95% CI)","PR-AUC","Brier","Cal. intercept","Cal. slope","ICI"],table2rows)+"\n\n"
table2 += "Stay-level test split: n = 184; 33 events. Five-fold OOF AUC: "+oof_auc+". Patient-grouped holdout AUC: "+group_test_auc+"; patient-grouped OOF AUC: "+group_oof_auc+". Pure-rounding integer score AUC: 0.618 (0.510–0.722). XGBoost and LightGBM calibration slopes were unstable and were not used as evidence of validated alternatives.\n\n"
A=hfa.loc['A_current_index_admission_exclusion']
B=hfa.loc['B_strict_time_exclude_uncertain']
C=hfa.loc['C_expanded_possible_new_onset']
decomp=iptw.iloc[1]
table3rows=[
 ["Adjusted logistic, patient-clustered SE",f"{A.adjusted_OR:.2f} ({A.adjusted_OR_lo:.2f}–{A.adjusted_OR_hi:.2f})",f"{A.adjusted_p:.4f}"],
 ["Measured-selection IPW, patient-clustered SE",f"{A.IPW_OR:.2f} ({A.IPW_OR_lo:.2f}–{A.IPW_OR_hi:.2f})",f"{A.IPW_p:.4f}"],
 ["Decompensated phenotype, IPW",f"{decomp.OR_iptw:.2f} ({decomp.lo:.2f}–{decomp.hi:.2f})",f"{decomp.p:.4f}"],
 ["Strict-time HF exclusion (B), adjusted",f"{B.adjusted_OR:.2f} ({B.adjusted_OR_lo:.2f}–{B.adjusted_OR_hi:.2f})",f"{B.adjusted_p:.4f}"],
 ["Expanded possible-new-onset (C), adjusted",f"{C.adjusted_OR:.2f} ({C.adjusted_OR_lo:.2f}–{C.adjusted_OR_hi:.2f})",f"{C.adjusted_p:.4f}"]]
table3="### Table 3. Operational SICM and 28-day mortality\n\n"+md_table(["Estimator","OR (95% CI)","P"],table3rows)+"\n\n"
table3 += "Adjusted RD was 12.7 pp (2.3 to 22.3). The current-specification patient-cluster full-procedure IPW bootstrap gave a 95% interval of 0.84–4.54 (300/300 successful replicates), demonstrating substantial uncertainty. ORs are adjusted for age, SOFA-24h, Charlson index, lactate, ventilation and AKI stage.\n\n"
def pp(v): return f"{100*v:+.2f}"
def medrow(label,r):
    return [label,f"{pp(r.TE)} pp ({pp(r.TE_lo)} to {pp(r.TE_hi)})",f"{pp(r.NDE)} pp ({pp(r.NDE_lo)} to {pp(r.NDE_hi)})",
            f"{pp(r.NIE)} pp ({pp(r.NIE_lo)} to {pp(r.NIE_hi)})",f"{100*r.PM:.1f}% ({100*r.PM_lo:.1f} to {100*r.PM_hi:.1f})"]
table4rows=[medrow("MIMIC / vasopressor initiation-up-titration",med.loc['vasopressor initiation/up-titration']),
            medrow("MIMIC / peak norepinephrine (log1p)",med.loc['peak norepinephrine (log1p)'])]
eicu=load("aim_eicu_V3_mediation.csv").set_index("effect")
table4rows.append(["eICU proxy / norepinephrine use",f"+1.9 pp (0.1 to 4.1)",f"+2.2 pp (0.2 to 4.2)","−0.32 pp (−0.40 to −0.01)","−16.8% (−62.0 to 5.5; suppression)"])
table4="### Table 4. Exploratory effect decomposition of SICM/proxy-exposure associations\n\n"+md_table(
 ["Cohort / candidate mediator","Total effect","Natural direct effect","Natural indirect effect","Proportion mediated (exploratory)"],table4rows)+"\n\n"
table4 += "Natural-effect labels are retained for transparency, but exposure–mediator order is not established. These are exploratory associational decompositions, not identified causal effects. MIMIC intervals use 2,000 bootstrap replicates; eICU uses 300.\n\n"
main_tables=table1+table2+table3+table4
old_main=(PKG/"Manuscript_SICM_BMC.md").read_text(encoding="utf-8")
declarations=old_main[old_main.index("## Declarations"):old_main.index("## References")]
references=old_main[old_main.index("## References"):old_main.index("Tables\n")]
figures="""## Figures

![Fig. 1](figures/Figure1.png)

Fig. 1 Corrected MIMIC-IV cohort derivation. Of 18,273 Sepsis-3 stays, 2,350 had troponin and LVEF records; 1,053 were excluded for age or ACS, leaving 1,297 stays. The current no-index-admission-HF cohort contained 613 stays (111 SICM; 502 reference), and the index-admission HF group contained 684 stays (355 SICM). No separate cardiac-surgery/transplant/VAD exclusion was implemented.

![Fig. 2](figures/Figure2.png)

Fig. 2 Model performance in the stay-level held-out test set (n = 184; 33 SICM): ROC, calibration and exploratory decision-curve analysis. These panels do not represent the patient-grouped or full-cohort OOF evaluations.

![Fig. 3](figures/Figure3.png)

Fig. 3 Exploratory XGBoost SHAP explanation fitted in all 613 stays (111 SICM; 502 reference). SHAP values do not establish biological mechanisms and are not coefficients of the primary LASSO model.

![Fig. 4](figures/Figure4.png)

Fig. 4 Haemodynamic analyses. (A) Exploratory effect decomposition in MIMIC (n = 613; 111 SICM) and the eICU proxy cohort (n = 6,006). Natural-effect labels do not establish causal mediation. (B) Unadjusted observed mortality by norepinephrine-dose group; confounding by indication precludes causal interpretation.

Supplementary materials: Supplements S1–S9, Supplementary Tables S1–S8, Supplementary Figures S1–S5, and the TRIPOD+AI checklist are provided separately.
"""
main_body += declarations+references+"Tables\n\n"+main_tables+figures
(PKG/"Manuscript_SICM_BMC.md").write_text(main_body,encoding="utf-8")
(PKG/"Tables_BMC.md").write_text("# Tables\n\nfor: Corrected Round-7 analysis\n\n"+main_tables,encoding="utf-8")
print("Main manuscript and Tables 1-4 rendered")
# Supplement S1-S9 and Supplementary Tables S1-S8.
elig=load("round7_eligibility.csv")
supp="# Supplementary Material\nfor: Corrected Round-7 post-hoc analysis\n\n"
supp += "Contents: S1 Eligibility | S2 eICU proxy | S3 strict features and timestamps | S4 imaging quality | S5 reporting checklist | S6 IPW diagnostics | S7 sensitivity analyses | S8 model and score | S9 HF timing, patient-level and exposure-timing analyses\n\n"
supp += "## Supplement S1. Eligibility sensitivity\n\nRestrictions use the corrected cohort. `first_icu_stay` is a MIMIC-derived flag and is not a substitute for patient-grouped validation.\n\n"+md_table(
 ["Restriction","n","SICM events","Adjusted OR (95% CI)","P","Test AUC"],
 [[r.restriction,int(r.n),int(r.events),r.adjusted_OR_CI,f"{r.p:.4f}",f"{r.test_AUC:.3f}"] for r in elig.itertuples()])+"\n\n"
supp += "## Supplement S2. eICU proxy cohort and construct limitations\n\nThe eICU-CRD proxy cohort contains 6,006 stays, 3,902 proxy-exposed stays and 462 norepinephrine users. Its 28-day in-hospital mortality proxy is 21.3%. eICU contains no structured LVEF and no post-discharge deaths and is not external validation. The current reproducibility package does not contain the complete upstream construction of `eicu_crd.temp_final_export`; this limitation is explicitly retained rather than implying independent rebuildability.\n\n"
supp += "## Supplement S3. Corrected feature and timestamp definitions\n\nACS was ascertained using ICD-9 410/411 and ICD-10 I21/I22. Qualifying low LVEF and decline follow-up were at/after suspected infection. NE, dobutamine, milrinone and invasive-ventilation starts were restricted to ICU 0–24 h; no pre-ICU start was included. AKI used KDIGO records in the same window. The locked model uses 44 candidates and excludes dobutamine/milrinone. No separate cardiac-surgery/transplant/VAD exclusion was implemented.\n\n"
supp += "## Supplement S4. Echocardiographic timing and baseline confirmation\n\n"
s=dn[dn.sicm.eq(1)]
supp += md_table(["Characteristic","Value"],[
 ["Current SICM events",len(s)],["Timestamped post-infection LVEF <50%",int(s.ef50_h.notna().sum())],
 ["Median time to first qualifying LVEF <50%, h (IQR)",f"{s.ef50_h.median():.1f} ({s.ef50_h.quantile(.25):.1f}–{s.ef50_h.quantile(.75):.1f})"],
 ["Decline-only cases",int((s.ef50_h.isna()).sum())],["Cases with ≥2 LVEF records",39],
 ["Any pre-ICU LVEF ≥50%",26],["Pre-ICU LVEF ≥50% within 7 days",7]])+"\n\n"
supp += "The 7-day window is explicit. Earlier normal LVEF does not establish that most cases were truly de novo.\n\n"
supp += "## Supplement S5. Reporting checklist\n\nThe separate TRIPOD+AI checklist should be read with the Round-7 corrections. Original split, patient-grouped validation, current model formula, HF timing, exposure timing and post-hoc status are reported. No formal PROBAST+AI domain-level rating was performed.\n\n"
smd=load("round7_ipw_balance_smd.csv")
supp += "## Supplement S6. Corrected IPW diagnostics\n\nThe corrected source population contains 14,980 stays: 1,297 evaluable and 13,683 non-evaluable. Propensity AUC was 0.713; truncated weights ranged from 0.20 to 3.40 and effective sample size was 876. IPW corrects measured selection only.\n\n"
supp += md_table(["Covariate","SMD before weighting","SMD after weighting"],
 [[r.covariate,f"{r.SMD_unweighted:.3f}",f"{r.SMD_weighted:.3f}"] for r in smd.itertuples()])+"\n\n"
supp += "## Supplement S7. Additional sensitivity analyses\n\n"
supp += md_table(["Analysis","Result"],[
 ["Excluding 2020–2022 block","n=569; 102 events; adjusted OR 1.65 (1.05–2.59), P=0.031"],
 ["Excluding 9 decline-only cases","n=604; 102 events; adjusted OR 1.74 (1.11–2.73), P=0.016; test AUC 0.576 (0.448–0.699)"],
 ["Multiple imputation (mice, m=20)","Adjusted OR 1.78 (1.15–2.76), consistent with median imputation"],
 ["LVEF recovery","13/26 recovered; median 44.1 h; 10 recoveries within 168 h; descriptive mortality 23.1% vs 38.5%"],
 ["Riley global shrinkage","Not met; required n≈2,825 versus available 613"],
 ["Weighted OOF in ICD-10-only cohort","OOF 0.666; IPW-weighted OOF 0.667"],
 ["Worst-subgroup recalibration","AKI 1 ICI 0.104 to 0.090"],
 ["Landmark transport","n=529; 40 later events; transport AUC 0.650; within-landmark refit AUC 0.527"],
 ["Unadjusted norepinephrine spline","Non-linearity χ²=12.32, df=3, P=0.0064; exploratory and confounded by indication"]])+"\n\n"
# Complete current penalised formula and pure-rounding score.
params=load("round7_primary_lasso_model_parameters.csv")
score=load("round7_score_card_full.csv")
supp += "## Supplement S8. Current penalised model and integer score\n\n"
supp += f"Development set: 429 stays/78 events; test set: 184/33. All 44 candidates have non-zero penalised coefficients. Selected C = {artifact['selected_C']:.8f}. Continuous predictors are median-imputed and z-standardised using development-set population SD; binary predictors remain 0/1. Intercept = {artifact['intercept']:.6f}.\n\n"
supp += md_table(["Predictor","Imputation median","Mean","Population SD","Penalised beta"],
 [[r.feature,"—" if pd.isna(r.train_imputation_median) else f"{r.train_imputation_median:.5g}",
   "—" if pd.isna(r.train_mean) else f"{r.train_mean:.5g}","—" if pd.isna(r.train_sd_population) else f"{r.train_sd_population:.5g}",
   f"{r.penalised_coefficient:.6f}"] for r in params.itertuples()])+"\n\n"
supp += "The pure-rounding score is shown below; the forced-sign alternative is not used for the headline score.\n\n"
supp += md_table(["Predictor","Mean","SD","Unpenalised std. coef.","Points (pure rounding)"],
 [[r.feature,"—" if pd.isna(r.mean) else f"{r.mean:.4g}","—" if pd.isna(r.sd) else f"{r.sd:.4g}",f"{r.coef_z:.3f}","—" if pd.isna(r.points_raw) else int(r.points_raw)] for r in score.itertuples()])+"\n\n"
supp += f"Pure-rounding test AUC was 0.618 (0.510–0.722); forced-sign AUC was {score_map.auc_points:.3f}. Score-to-risk mapping was logit(p) = {score_map.map_intercept:.3f} + {score_map.map_slope:.3f} × points. It is not validated for clinical use.\n\n"
supp += "## Supplement S9. HF timing, patient-level and exposure-timing analyses\n\n"
supp += md_table(["HF record classification","ICU stays","SICM events"],
 [[label,int(time_groups.loc[key,'n']),int(time_groups.loc[key,'events'])] for key,label in [
  ('documented_prior_hf','Documented earlier-admission HF'),('index_first_hf_time_uncertain','Index-first/time-uncertain HF'),('no_hf_diagnosis_record','No available HF diagnosis record')]])+"\n\n"
labels={"A_current_index_admission_exclusion":"A: current index-admission exclusion","B_strict_time_exclude_uncertain":"B: strict-time exclusion","C_expanded_possible_new_onset":"C: expanded possible new-onset"}
supp += md_table(["Scheme","n / patients","SICM","Deaths SICM/reference","Adjusted OR (95% CI)","IPW OR (95% CI)","Adjusted RD, pp (95% CI)","OOF AUC (95% CI)"],
 [[label,f"{int(x.n)} / {int(x.patients)}",int(x.sicm_events),f"{int(x.deaths_sicm)} / {int(x.deaths_reference)}",ci(x,'adjusted_OR'),ci(x,'IPW_OR'),
   f"{100*x.adjusted_RD:.1f} ({100*x.adjusted_RD_lo:.1f} to {100*x.adjusted_RD_hi:.1f})",auc3(hfp.loc[key])] for key,label in labels.items() for x in [hfa.loc[key]]])+"\n\n"
supp += md_table(["Patient-grouped evaluation","n / patients","Events","AUC (95% CI)","PR-AUC","Brier","Cal. intercept","Cal. slope","ICI"],
 [[label,f"{int(x.n)} / {int(x.patients)}",int(x.events),auc3(x),f"{x.PR_AUC:.3f}",f"{x.Brier:.3f}",f"{x.calibration_intercept:.3f}",f"{x.calibration_slope:.3f}",f"{x.ICI:.3f}"]
  for key,label in [('patient_grouped_holdout','Held-out'),('patient_grouped_oof','Five-fold OOF')] for x in [valid.loc[key]]])+"\n\n"
supp += md_table(["Mortality analysis","n / patients","Exposure events","Deaths","HR (95% CI)","P"],
 [[label,f"{int(x.n)} / {int(x.patients)}",int(x.sicm_events),int(x.deaths),ci(x,'HR'),f"{x.p:.4g}"]
  for key,label in [('time_dependent_cox','Time-dependent SICM'),('landmark_24h_early_sicm','24-h landmark, early SICM')] for x in [time.loc[key]]])+"\n\n"
supp += "These HRs use age, sex and Charlson index as the reduced baseline adjustment set and are not numerically equivalent to the fixed-exposure OR. No new causal mediation analysis was performed because exposure–mediator temporal ordering is not established.\n\n"
s1rows=[
 ["MIMIC descriptive cohort","Corrected ACS and post-infection imaging","1,297","466 SICM","Table 1; Figure 1"],
 ["Current no-index-HF cohort","No index-admission HF diagnosis","613","111 SICM; 502 reference","Primary model, mortality and decomposition"],
 ["Development split","Original stratified split","429","78 SICM","Model fitting"],
 ["Held-out test split","Original stratified split","184","33 SICM","Test performance, DCA and exploratory threshold"],
 ["Stay-level five-fold OOF","All current stays","613","111 SICM","Internal validation"],
 ["Patient-grouped holdout","Patient-grouped split","184 stays / 182 patients","29 SICM","Supplement S9"],
 ["HF strict-time cohort (B)","No earlier- or index-admission HF","581","104 SICM","HF timing sensitivity"],
 ["Expanded possible-new-onset cohort (C)","No earlier-admission HF; index-first included","1,017","330 SICM","HF timing sensitivity"],
 ["ICD-10-only HF-ascertainment cohort","Legacy has_hf=0 cohort","1,018","318 SICM","Coding and subgroup sensitivity"],
 ["Selection source population","Corrected age/ACS eligibility","14,980","1,297 evaluable","Measured-selection IPW"],
 ["Time-dependent Cox","Valid complete onset and death times","611","110 exposure events; 231 deaths","Exposure-timing sensitivity"],
 ["24-h mortality landmark","Alive beyond 24 h","598","61 early SICM; 218 deaths","Landmark sensitivity"],
 ["eICU proxy cohort","Troponin I available; no LVEF","6,006","3,902 proxy exposed","Exploratory proxy pathway only"]]
supp += "### Supplementary Table S1. Analysis-set register\n\n"+md_table(["Analysis set","Derivation / relationship","n","Events / evaluable n","Purpose"],s1rows)+"\n\nRows are not mutually exclusive and must not be summed.\n\n"
legacy_s=legacy[legacy.sicm.eq(1)];legacy_r=legacy[legacy.sicm.eq(0)]
s2rows=[["Analysis cohort, n",f"{len(legacy):,}",f"{n_dn:,}"],["SICM events, n (%)",f"{int(legacy.sicm.sum())} ({100*legacy.sicm.mean():.1f})",f"{n_events} ({100*dn.sicm.mean():.1f})"],["28-day mortality, SICM",pct(legacy_s.mort_28d.mean(),1),pct(deaths_s,n_events)],["28-day mortality, reference",pct(legacy_r.mort_28d.mean(),1),pct(deaths_r,n_ref)],["Adjusted OR (95% CI)","1.15 (0.86–1.55)","1.78 (1.15–2.76)"],["IPW OR (95% CI)","1.12 (0.82–1.54)","1.94 (1.14–3.29)"]]
supp += "### Supplementary Table S2. Coding-era sensitivity of HF ascertainment\n\n"+md_table(["Metric","ICD-10 I50 only (legacy)","Era-spanning (current)"],s2rows)+"\n\nThese analyses address coding-era classification, not complete HF timing or clinical de novo status.\n\n"
old_supp=(PKG/"Supplement_SICM_BMC.md").read_text(encoding="utf-8")
s3=old_supp[old_supp.index("### Supplementary Table S3"):old_supp.index("### Supplementary Table S4")]
s3=s3.replace("Present study | Retrospective; MIMIC-IV-Echo; era-spanning heart-failure ascertainment; IPW for imaging selection; competing-risk incidence | 789 (215 SICM, 27.2%) | Prior myocardial infarction, heart rate, haemoglobin, temperature, glucose, respiratory rate (21-variable model) | AUC 0.731 (95% CI 0.655–0.795); out-of-fold 0.677; bedside score 0.719 | Troponin is definitional and therefore excluded from the candidate set; heart failure ascertained across ICD-9 and ICD-10 eras; imaging selection modelled by IPW; incidence estimated with death as a competing event",
"Present study, Round-7 | Retrospective; corrected MIMIC-IV-Echo; era-spanning HF and ACS ascertainment; measured-selection IPW | 613 (111 SICM, 18.1%) | All 44 candidates retained | Test AUC 0.634; stay-level OOF 0.616; patient-grouped OOF 0.590; pure-rounding score 0.618 | Post-infection imaging required; ICD-9/10 ACS excluded; strict treatment starts; exploratory only")
supp += s3
supp += "### Supplementary Table S4. Echocardiographic timing\n\n"+md_table(["Characteristic","Value"],[["SICM cases",111],["Post-infection LVEF <50 with timestamp",102],["Median time, h (IQR)","18.3 (11.6–47.6)"],["Decline-only cases",9],["Cases with ≥2 LVEF records",39],["Any pre-ICU LVEF ≥50",26],["Pre-ICU LVEF ≥50 within 7 days",7]])+"\n\n"
sens=load("round7_sensitivity_phenotypes_mortality.csv")
supp += "### Supplementary Table S5. Sensitivity phenotypes\n\n"+md_table(["Phenotype","Events","Adjusted OR (95% CI)","P"],[[r.phenotype,int(r.events),f"{r.OR:.2f} ({r.lo:.2f}–{r.hi:.2f})",f"{r.p:.4f}"] for r in sens.itertuples()])+"\n\n"
sub=load("round7_subgroup_heterogeneity.csv")
supp += "### Supplementary Table S6. Era and subgroup validation\n\n"+md_table(["Dimension","Subgroup","n","Events","AUC (95% CI)","P","BH-FDR q"],[[r.dim,r.subgroup,int(r.n),int(r.events),"" if pd.isna(r.AUC) else f"{r.AUC:.3f} ({r.lo:.3f}–{r.hi:.3f})","" if pd.isna(r.p_vs_overall) else f"{r.p_vs_overall:.3f}","" if pd.isna(r.q_FDR) else f"{r.q_FDR:.3f}"] for r in sub.itertuples()])+"\n\n"
th=metrics.loc["Test threshold nearest sensitivity 0.65 (exploratory)"]
tp=round(th.sensitivity*33);fn=33-tp;tn=round(th.specificity*151);fp=151-tn
supp += "### Supplementary Table S7. Exploratory test-set operating point\n\n"+md_table(["Metric","Value"],[["Threshold",f"{th.threshold:.6f}"],["SICM / non-SICM","33 / 151"],["True positives",tp],["False positives",fp],["False negatives",fn],["True negatives",tn],["Sensitivity",f"{100*th.sensitivity:.1f}%"],["Specificity",f"{100*th.specificity:.1f}%"],["Patients flagged",f"{int(th.flagged_n)}/184 ({100*th.flagged_n/184:.1f}%)"]])+"\n\nThe threshold was selected and evaluated in the same test set and is not a clinical triage rule.\n\n"
fg=load("round7_comprisk_finegray.csv");rcs=load("round7_rcs.csv").iloc[0]
supp += "### Supplementary Table S8. Competing-risk and spline analyses\n\n"+md_table(["Analysis","Estimate","95% CI / P"],[["Aalen–Johansen SICM at 28 days","17.9%","naive 1−KM 19.3%"],["Lactate, Fine–Gray sHR",f"{fg.iloc[0].HR:.3f}",f"{fg.iloc[0].lo:.3f}–{fg.iloc[0].hi:.3f}; P={fg.iloc[0].p:.2f}"],["SOFA, Fine–Gray sHR",f"{fg.iloc[1].HR:.3f}",f"{fg.iloc[1].lo:.3f}–{fg.iloc[1].hi:.3f}; P={fg.iloc[1].p:.2f}"],["Age, Fine–Gray sHR",f"{fg.iloc[2].HR:.3f}",f"{fg.iloc[2].lo:.3f}–{fg.iloc[2].hi:.3f}; P={fg.iloc[2].p:.2f}"],["Norepinephrine RCS non-linearity",f"χ²={rcs.chi2:.2f}, df=3",f"P={rcs.p:.4f}; unadjusted"]])+"\n\n"
supp += """![Supplementary Figure S1](supplementary_figures/FigureS1.png)

### Supplementary Figure S1. eICU cohort flow

The eICU proxy cohort (n = 6,006; 3,902 proxy exposed) is not external validation of the imaging phenotype or prediction model.

![Supplementary Figure S2](supplementary_figures/FigureS2.png)

### Supplementary Figure S2. Corrected IPW balance diagnostics

Evaluable (n = 1,297) versus non-evaluable (n = 13,683) stays in the corrected source population (n = 14,980).

![Supplementary Figure S3](supplementary_figures/FigureS3.png)

### Supplementary Figure S3. Subgroup discrimination

ICD-10-only HF-ascertainment cohort (n = 1,018; 318 SICM), overall OOF AUC 0.666. Subgroup analyses are descriptive and are not interaction tests.

![Supplementary Figure S4](supplementary_figures/FigureS4.png)

### Supplementary Figure S4. Bedside score and landmark ROC

(A) Test set (n = 184; 33 SICM). (B) Landmark cohort (n = 529; 40 later events): frozen transport AUC 0.650 and within-landmark refit AUC 0.527.

![Supplementary Figure S5](supplementary_figures/FigureS5.png)

### Supplementary Figure S5. Cumulative incidence

Current cohort (n = 613; 110 timestamped SICM events), with death as a competing event. The 28-day Aalen–Johansen estimate was 17.9% (naive 1−KM 19.3%).
"""
(PKG/"Supplement_SICM_BMC.md").write_text(supp,encoding="utf-8")
print("Supplement rendered")
