# -*- coding: utf-8 -*-
"""Prepare a publication-oriented Round-7 manuscript without changing locked results."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PKG=ROOT/"投稿包_BMC"
current=(PKG/"Manuscript_SICM_BMC.md").read_text(encoding="utf-8")
tables=current[current.index("### Table 1"):current.index("## Figures")]
figures=current[current.index("## Figures"):]
declarations=current[current.index("## Declarations"):current.index("## References")]
references=current[current.index("## References"):current.index("Tables\n")]

header="""# Left Ventricular Systolic Dysfunction in Sepsis without Index-Admission Heart Failure Diagnosis: 24-h Risk Recognition and 28-day Mortality: a retrospective development and internal validation study

Running title: Corrected risk recognition of septic left ventricular dysfunction

Article type: Original Research

Authors: [Author names, degrees and ORCID iDs — to be completed at submission]

Affiliations: [Institutional affiliations to be completed at submission]

Corresponding author: [Name, postal address, telephone, e-mail and ORCID to be completed at submission]

Reporting: TRIPOD+AI-, STROBE- and AGReMA-informed reporting. Round-7 is a post-hoc corrected analysis and is not represented as the original prespecified analysis.

"""
abstract="""## Abstract

Background Sepsis-induced cardiomyopathy (SICM) is sensitive to heart-failure (HF) ascertainment, acute coronary syndrome (ACS) exclusion and imaging timestamps. We reassessed an operational SICM phenotype after a data-chain audit identified inconsistencies in these upstream elements.

Methods In this post-hoc corrected MIMIC-IV/MIMIC-IV-Echo study, ACS was excluded using ICD-9 410/411 and ICD-10 I21/I22. Qualifying low LVEF and decline follow-up were required at/after suspected infection, and treatment starts were restricted to ICU 0–24 h. The current no-index-admission-HF cohort contained 613 stays and 111 SICM events. A 44-candidate LASSO-logistic model was evaluated using a stay-level split, out-of-fold (OOF) validation and patient-grouped validation. Mortality was analysed using covariate adjustment, measured-selection inverse probability weighting (IPW), HF-timing alternatives and exposure-time-sensitive models.

Results Twenty-eight-day mortality was 47.7% (53/111) with SICM versus 35.9% (180/502) without it. The adjusted OR was 1.78 (95% CI 1.15–2.76) and patient-clustered IPW OR 1.94 (1.14–3.29). Test AUC was 0.634 (0.530–0.738), stay-level OOF AUC 0.616 (0.556–0.672), and patient-grouped OOF AUC 0.590 (0.530–0.649). Strict-time HF exclusion yielded an adjusted OR of 1.77 (1.12–2.79), whereas expanded possible-new-onset classification attenuated the estimate to 1.09 (0.82–1.45). Time-dependent Cox and 24-h landmark HRs were 1.96 (1.42–2.70) and 2.13 (1.45–3.15), respectively.

Conclusions Corrected data supported an association between operational SICM and higher 28-day mortality, but discrimination was limited and calibration unstable. Estimates were sensitive to HF classification. The model, integer score and test-set operating point are exploratory and unsuitable for clinical triage. Multicentre imaging validation and better pre-sepsis phenotyping are required.

Keywords: ejection fraction; heart failure; MIMIC-IV; mortality; risk-recognition model; sepsis-induced cardiomyopathy

"""
intro="""## 1. Introduction

Sepsis-induced myocardial depression was described decades ago as a reversible component of septic shock [1]. Subsequent studies have linked inflammatory, mitochondrial, calcium-handling and microcirculatory abnormalities to sepsis-associated cardiac dysfunction [2–5]. Nevertheless, the reported incidence and prognostic meaning of SICM vary widely [6–13]. Part of this variation reflects biological heterogeneity, but part is produced by operational choices: ejection-fraction thresholds, troponin requirements, imaging timing, handling of pre-existing HF and selection of patients who undergo echocardiography [8–13].

These choices also affect prediction research. A model trained on one operational phenotype cannot be assumed to transport to another definition, particularly when troponin and imaging are used to define the outcome. Moreover, selective echocardiography creates a selected population in which disease severity and survival can influence imaging availability and outcome ascertainment. Discrimination within such a cohort therefore answers a different question from population-wide screening performance.

Administrative data introduce additional vulnerabilities. MIMIC-IV spans ICD-9 and ICD-10 eras, so HF and ACS cannot be ascertained adequately using ICD-10 codes alone. Discharge diagnoses also do not establish when HF began. A patient first coded with HF during the index admission may have incident HF, previously undocumented HF or an unrelated reason for coding. Finally, an ICU time window does not automatically guarantee that a treatment or echocardiogram occurred after sepsis onset or even after ICU admission.

Our initial analysis showed that numerical outputs can remain internally consistent while upstream definitions remain imperfect. The present Round-7 analysis therefore started with data-chain correction rather than further model tuning. We aimed to (1) construct a corrected, explicitly operational SICM cohort; (2) estimate its association with 28-day mortality using measured-selection weighting and exposure-time sensitivity analyses; and (3) assess 0–24-h risk recognition using stay-level and patient-level validation. We deliberately preserved uncertainty and did not select definitions, models or thresholds by significance or apparent performance.
"""
methods1="""## 2. Methods

### 2.1 Study design, data sources and reporting

This was a retrospective, single-database development and internal-validation study using MIMIC-IV v3.1 and structured measurements from MIMIC-IV-Echo [23,24]. Sepsis-3 was identified from suspected infection and SOFA≥2 [25]. The analysis was post hoc, prompted by the data-chain audit, and is not described as prospectively registered. Reporting was guided by TRIPOD+AI and STROBE [20,21], while pathway reporting was informed by AGReMA [41]. PROBAST+AI was used as a methodological checklist, not to assign a formal risk-of-bias rating [22].

Of 18,273 Sepsis-3 ICU stays, 2,350 had at least one troponin measurement and one structured LVEF-documented echocardiogram. Age and ACS exclusions removed 1,053 stays (117 for age and 989 for ACS, with overlap), leaving 1,297 stays in 1,271 patients. The source population used to model imaging evaluability contained 14,980 stays after the same age and ACS corrections, of whom 1,297 were evaluable and 13,683 were not.

### 2.2 ACS and HF ascertainment

ACS was identified from index-admission diagnoses using ICD-9 410.x/411.x and ICD-10 I21.x/I22.x. This corrected the earlier ICD-10-only implementation. No separate exclusion for prior cardiac surgery, transplantation or ventricular assist device was applied because no auditable code list was available; such an exclusion is not claimed.

The current HF comparison used the Charlson HF comorbidity component (Quan algorithm) applied across coding eras [27]. Index-admission HF was distinguished from HF documented in an earlier hospital admission. The descriptive cohort was divided into current no-index-admission-HF stays (n=613) and index-admission-HF stays (n=684). For sensitivity analyses, index-first HF was treated as time-uncertain rather than automatically classified as either prevalent or incident disease. Absence of an HF code was not interpreted as proof of no clinical HF.

### 2.3 Operational SICM phenotype

Operational SICM required all of the following: (i) no index-admission HF diagnosis for the current analysis cohort; (ii) post-suspected-infection LVEF <50% or a ≥10-percentage-point decline from the first ICU echocardiogram, with qualifying follow-up also at/after suspected infection; (iii) peak ICU troponin T >0.014 ng/mL; and (iv) no ACS by the corrected ICD-9/10 rule. Troponin was measured with itemid 51003 and reported to two decimal places, making >0.014 ng/mL equivalent to ≥0.02 ng/mL in this dataset. The threshold was operational, informed by the high-sensitivity TnT 99th-percentile upper reference limit [28,29], but assay-generation drift could not be modelled.

This definition yielded 466 events in the descriptive cohort and 111 in the current no-index-HF cohort. Nine events qualified by decline alone. Only seven cases had LVEF ≥50% within seven days before ICU; 26 had any pre-ICU LVEF ≥50%. Consequently, “new-onset” denotes an operational record-based phenotype, not independently confirmed de novo dysfunction.
"""
methods2="""### 2.4 Predictors and preprocessing

Candidate predictors were restricted to routine variables available within ICU 0–24 h. Troponin was excluded because it is part of the outcome definition. Treatment starts for norepinephrine, dobutamine, milrinone and invasive ventilation were required to be within ICU 0–24 h, with both lower and upper time bounds. AKI stage used KDIGO measurements within the same interval. Charlson comorbidities were treated as baseline administrative comorbidities, while severity variables were treated as early ICU measurements rather than guaranteed pre-exposure covariates.

Continuous variables were median-imputed and standardised using development-set statistics; binary variables were unscaled. Lactate and bilirubin missingness indicators were included. Albumin was excluded because of extensive missingness. The locked feature set contained 44 candidates; dobutamine and milrinone were not used by the final model. No test-set imputation, variable selection or model selection was performed.

### 2.5 Model development and evaluation

The primary model was LASSO-penalised logistic regression with ten-fold cross-validated penalty selection [30,31]. The original stratified split contained 429 development stays (78 events) and 184 test stays (33 events). A five-fold OOF analysis repeated the complete preprocessing and fitting process in all 613 stays. XGBoost and LightGBM served as comparators with the original fixed hyperparameters and Platt calibration; they were not used to choose a winner.

Performance was summarised using AUC, precision–recall AUC, Brier score, calibration intercept and slope, and integrated calibration index [32,33]. Test-set and OOF confidence intervals were obtained by stratified resampling. Decision-curve analysis was exploratory [34]. The operating point was selected within the test set to give sensitivity closest to 0.65 and is not a validated triage threshold. The integer score used pure coefficient rounding; a forced-sign variant is retained only for provenance.

Additional analyses used a patient-grouped holdout and patient-grouped nested cross-validation. All stays from one patient were confined to the same outer and inner partitions, and confidence intervals were obtained by resampling patients. These analyses were post hoc and do not constitute external validation.

### 2.6 Mortality association and measured-selection weighting

The adjusted logistic model included age, SOFA-24h, Charlson index, lactate, invasive ventilation and AKI stage. Because some SICM events occurred before 24 h, these variables were not assumed to be entirely pre-exposure. The model was therefore interpreted as an adjusted association, not as a fully identified causal effect.

IPW was used to address measured differences in echocardiographic evaluability [35]. The propensity model was fitted in the corrected source population using age, sex, SOFA-24h, early vital signs, selected laboratory measurements, ventilation and Charlson index. Stabilised weights were truncated at the 1st and 99th percentiles. Outcome-model standard errors were clustered by patient. A current-specification full-procedure bootstrap repeated propensity fitting and outcome estimation in 300 patient-cluster replicates. Risk differences were estimated by g-computation with 500 patient-cluster bootstrap replicates.

### 2.7 HF timing and exposure-time sensitivity analyses

Three HF classification schemes were evaluated without selecting a preferred result by significance or AUC. Scheme A reproduced current index-admission exclusion; scheme B excluded both earlier-admission HF and index-first/time-uncertain HF; scheme C excluded earlier-admission HF but allowed index-first HF as possible new-onset disease. These schemes assess record-classification sensitivity, not the truth of de novo dysfunction.

For exposure-time sensitivity, SICM onset was defined as the latest of suspected infection, qualifying imaging and first positive troponin. Time-dependent Cox models used start–stop follow-up, and 24-h landmark models included patients alive beyond 24 h. Both used age, sex and Charlson index as a reduced baseline adjustment set and patient-clustered robust variance. Their HRs are not numerically equivalent to the fixed-exposure logistic OR.

### 2.8 Pathway analysis and other sensitivity analyses

Exploratory g-computation estimated total, direct and indirect associations for vasopressor initiation/up-titration and peak norepinephrine-equivalent dose [37–40]. Natural-effect labels were retained for comparability, but exposure–mediator temporal ordering was not guaranteed. Accordingly, these estimates were treated as exploratory effect decomposition, not identified causal mediation. The eICU analysis remained a proxy-pathway analysis because eICU lacks structured LVEF and post-discharge mortality; it was not external validation.

Sensitivity analyses included HF coding-era comparison, first-stay and ≥24-h restrictions, alternative LVEF definitions, exclusion of decline-only cases and the 2020–2022 block, multiple imputation, competing-risk cumulative incidence and Fine–Gray models, landmark transport, and descriptive LVEF recovery. All random procedures used seed 20260101. Python 3.12 and R 4.6 were used.
"""
results1="""## 3. Results

### 3.1 Cohort and clinical characteristics

The corrected descriptive cohort contained 1,297 stays, of which 466 met the operational SICM definition. The current no-index-HF cohort contained 613 stays: 111 with SICM and 502 references. The index-admission-HF group contained 684 stays, including 355 events. Table 1 shows that current SICM cases had more tachycardia, higher troponin and longer ICU stays than references, but lower prior myocardial infarction and Charlson comorbidity indices than the index-admission-HF group. These descriptive contrasts were not used as evidence of biological mechanisms.

The first qualifying LVEF <50% occurred at a median of 18.3 h after ICU admission (IQR 11.6–47.6), based on 102 timestamped cases; nine additional cases qualified by decline alone. Thirty-nine cases had at least two LVEF records. Twenty-eight-day deaths numbered 53/111 (47.7%) in the SICM group and 180/502 (35.9%) in the reference group.

### 3.2 Mortality association

The adjusted OR for 28-day mortality was 1.78 (95% CI 1.15–2.76; P=0.0097), and the patient-clustered IPW OR was 1.94 (1.14–3.29; P=0.0145). The adjusted risk difference was 12.7 percentage points (95% CI 2.3 to 22.3). Multiple imputation produced a nearly identical estimate (OR 1.78, 1.15–2.76). The current-specification, patient-clustered full-procedure IPW bootstrap yielded an interval of 0.84–4.54, demonstrating that the weighted estimate remained uncertain despite the positive point estimate.

Strict-time HF exclusion (scheme B) produced an adjusted OR of 1.77 (1.12–2.79), close to the current scheme. In contrast, expanded possible-new-onset classification (scheme C) attenuated the estimate to 1.09 (0.82–1.45). Thus, the positive association was sensitive to whether index-first HF was admitted as possible new-onset disease. These estimates do not identify which classification is clinically correct.

Time-dependent Cox analysis included 611 stays, 110 exposure events and 231 deaths, giving an HR of 1.96 (1.42–2.70). The 24-h landmark analysis included 598 stays, 61 early SICM cases and 218 deaths, giving an HR of 2.13 (1.45–3.15). Both models used the reduced baseline adjustment set and addressed exposure timing more explicitly than the fixed-exposure logistic model.

### 3.3 Model discrimination and calibration

The primary LASSO model achieved a test AUC of 0.634 (0.530–0.738), PR-AUC 0.314 and Brier score 0.153. Calibration intercept was −0.102, slope 0.371 and ICI 0.084. Stay-level OOF AUC was 0.616 (0.556–0.672). For comparison, age+SOFA had a test AUC of 0.564, the 1-SE LASSO variant 0.581, XGBoost 0.638 and LightGBM 0.612. The comparator calibration slopes were markedly unstable and were not interpreted as evidence of robust alternatives.

Patient-grouped evaluation further reduced discrimination: holdout AUC was 0.539 (0.435–0.639) and nested OOF AUC 0.590 (0.530–0.649). All 44 candidates had non-zero penalised coefficients, corresponding to approximately 2.5 events per candidate. The global-shrinkage criterion was not met, with an estimated required sample size of approximately 2,825. Coefficient signs should not be interpreted as protective or harmful effects because of collinearity, selection and model instability.

The pure-rounding integer score had a test AUC of 0.618 (0.510–0.722). The forced-sign variant had a numerically different AUC and is not used as the headline score. The post-hoc threshold was 0.157735: sensitivity was 63.6%, specificity 61.6%, and 79/184 patients were flagged. Because this threshold was selected and evaluated in the same test set, these operating characteristics are exploratory and not a basis for clinical triage.

### 3.4 Other sensitivity analyses

Excluding the 2020–2022 block gave an adjusted OR of 1.65 (1.05–2.59), and excluding nine decline-only cases gave 1.74 (1.11–2.73). The LVEF <40% phenotype had an adjusted OR of 1.63 (0.92–2.89), the ≥15-point decline phenotype 1.77 (1.14–2.74), and imaging-only phenotype 1.77 (1.15–2.73). The decompensated phenotype was imprecise (adjusted OR 1.48, 0.82–2.65).

The 28-day Aalen–Johansen cumulative incidence was 17.9%, compared with 19.3% using naive 1−Kaplan–Meier. Fine–Gray estimates for lactate, SOFA and age were imprecise and not significant. Thirteen of 26 cases with follow-up imaging recovered to LVEF ≥50%, at a median of 44.1 h; 10 recoveries occurred within 168 h. Mortality in this selected recovery subset was 23.1% versus 38.5% among non-recovered cases, a descriptive comparison that does not support treatment or prognosis claims.

### 3.5 Pathway findings

Exploratory MIMIC decomposition estimated a total effect of +12.7 percentage points for vasopressor initiation/up-titration, with NIE −0.18 points (−1.52 to 1.01). For peak norepinephrine, the NIE was +0.34 points (−0.50 to 1.33). Neither indirect interval suggested a positive mediated pathway, but uncertain exposure–mediator ordering prevents a conclusion of no mediation. The eICU proxy analysis showed a small negative indirect association consistent with suppression, not independent confirmation of the MIMIC imaging phenotype.

Peak norepinephrine dose showed an unadjusted non-linear association with mortality (χ²=12.32, df=3, P=0.0064). This gradient is vulnerable to confounding by indication and must not be interpreted as evidence of drug harm.
"""
discussion1="""## 4. Discussion

### 4.1 Principal findings

Three findings emerge from the corrected analysis. First, operational SICM remained associated with higher 28-day mortality after covariate adjustment and measured-selection weighting. Second, the association was sensitive to HF classification, although it persisted under strict-time exclusion and exposure-time-sensitive models. Third, risk-recognition performance was limited and calibration unstable, particularly under patient-grouped evaluation. These findings support an association within a selected operational cohort, not a clinically deployable prediction rule.

The corrected data therefore separate two questions that are sometimes conflated: whether an operational phenotype has prognostic associations and whether early routine variables can identify that phenotype reliably. A positive mortality association does not imply useful discrimination. In this study, the same data supported the former more strongly than the latter.

### 4.2 Importance of upstream data-chain correction

The corrections materially changed cohort composition and performance. Including ICD-9 ACS codes was essential because restricting exclusion to ICD-10 would retain many ICD-9-era ACS admissions. Requiring qualifying imaging after suspected infection also changed the phenotype: an abnormal LVEF recorded before suspected infection cannot simply be labelled sepsis-induced. Similarly, treatment starts before ICU cannot be described as treatment initiated within ICU 0–24 h. These are not cosmetic distinctions; they alter the estimand and the cases being modelled.

The present estimates should not be compared with the earlier cohort as if only sample size had changed. ACS exclusion, imaging eligibility and treatment definitions all changed simultaneously. The analysis therefore represents a corrected post-hoc version rather than a sensitivity estimate on an identical population. This distinction is central to interpreting the work and to avoiding retrospective claims of prespecification.

### 4.3 HF timing and the meaning of “new-onset”

Strict-time exclusion produced an estimate close to the current classification, whereas expanded possible-new-onset classification attenuated the association. This attenuation is compatible with contamination of the expanded group by patients with prevalent but previously uncoded dysfunction, but other explanations are possible, including differences in severity, treatment or coding practice. The analyses cannot determine which patients truly had de novo dysfunction.

Only seven cases had normal LVEF within seven days before ICU. Accordingly, the operational label should not be equated with proven previously normal systolic function. Future studies need protocolised pre-sepsis imaging or carefully validated historical assessments, rather than relying on absence of HF codes alone. The current title and text deliberately refer to absence of an index-admission diagnosis rather than absence of clinical HF.

### 4.4 Interpretation of model performance

The primary test and OOF estimates were modest, and patient-grouped estimates were lower. This pattern does not support claims of a robust early-warning tool. All 44 candidates were retained, with approximately 2.5 events per candidate, and the global-shrinkage criterion was not met. The large penalty-selection value and unstable calibration are additional reasons for caution. Comparator models did not rescue performance and displayed marked calibration instability.

The integer score has the same limitations and is provided for reproducibility, not deployment. Likewise, the test-set threshold and decision curve are exploratory. Selecting a threshold and reporting its sensitivity and specificity on the same 184-stay test set can produce optimistic operating characteristics. External imaging cohorts with prespecified thresholds would be required before any clinical interpretation.
"""
discussion2="""### 4.5 Mortality association and selection

The adjusted, IPW and exposure-time-sensitive estimates were directionally consistent, but several qualifications remain. Echocardiography was available in only 1,297 of 14,980 eligible stays. Evaluable patients had higher observed early mortality than non-evaluable patients (36.0% versus 17.7%) and higher median SOFA scores (7 versus 5). Although the propensity model and truncation improved measured balance, residual clinical selection and positivity limitations cannot be excluded.

The wide full-procedure bootstrap interval is particularly important: it shows that reliance on a single weighted point estimate would overstate certainty. Moreover, some adjustment variables may have been measured after SICM onset, so post-exposure adjustment remains possible. The results should therefore be described as associations within an operational selected cohort, not as proof that SICM causes excess mortality.

### 4.6 Pathway interpretation

No positive indirect association was observed through vasopressor initiation/up-titration or peak norepinephrine dose. However, these analyses cannot establish a direct biological pathway. Mediators were aggregated after sepsis rather than after complete SICM onset, unmeasured confounding is possible, and the eICU proxy has a different construct and incomplete post-discharge mortality. The appropriate conclusion is that no positive indirect association was detected, not that vasopressors play no mediating role.

The unadjusted dose–mortality gradient is similarly vulnerable to confounding by indication. Patients requiring higher norepinephrine doses are generally more severely ill. These data do not justify attributing mortality to norepinephrine itself or altering vasopressor practice.

### 4.7 Clinical implications and future work

The immediate implication is methodological: imaging and coding definitions must be resolved before model development. A clinically useful next step would be a prospective multicentre cohort with protocolised early echocardiography, documented pre-sepsis LVEF where available, assay-specific troponin handling and explicit exposure timestamps. Such a cohort should evaluate both the operational mortality association and the generalisability of risk recognition.

If a revised model is developed, it should use fewer candidate parameters, preserve patient-level separation and undergo external validation with calibration assessment. The current 44-variable model, integer score and threshold should not be carried forward as a bedside tool.

### 4.8 Limitations

This study is limited by its retrospective single-centre design and selected imaging cohort. Pre-sepsis LVEF was unavailable for most patients, and HF coding is imperfectly timed and validated. ACS codes identify documented diagnoses rather than excluding every possible ischaemic event. No separate cardiac-surgery/transplant/VAD exclusion was implemented. Structured LVEF was not core-laboratory remeasured, and assay-generation drift could not be modelled.

Some covariates may have been measured after exposure, and unmeasured imaging selection, residual confounding and guarantee-time bias cannot be excluded. Repeat ICU stays, limited events and all-44-variable retention increase uncertainty. The threshold was selected and evaluated in the same test set. eICU provides no independent imaging validation. Finally, the corrected analysis was post hoc, and earlier versions must not be used to imply prospective registration.

## 5. Conclusions

Corrected operational SICM was associated with higher 28-day mortality, but early discrimination was limited and estimates varied with HF classification. The current model, integer score and test-set threshold are exploratory and unsuitable for clinical triage. Prospective multicentre imaging cohorts with protocolised timing and pre-sepsis phenotyping are required before model revision or clinical use.
"""
abbreviations="""## Abbreviations

ACS, acute coronary syndrome; AUC, area under the ROC curve; CI, confidence interval; DCA, decision-curve analysis; HF, heart failure; HR, hazard ratio; ICI, integrated calibration index; IPW, inverse probability weighting; LASSO, least absolute shrinkage and selection operator; LVEF, left ventricular ejection fraction; NDE/NIE, natural direct/indirect effect; OOF, out-of-fold; OR, odds ratio; PR-AUC, area under the precision–recall curve; RCS, restricted cubic spline; SHAP, Shapley additive explanations; SICM, sepsis-induced cardiomyopathy; SOFA, sequential organ failure assessment; VAD, ventricular assist device.

"""
manuscript=header+abstract+intro+methods1+methods2+results1+discussion1+discussion2+abbreviations+declarations+references+"Tables\n\n"+tables+figures
(PKG/"Manuscript_SICM_BMC.md").write_text(manuscript,encoding="utf-8")
print("Publication-oriented manuscript written; words:",len(manuscript.split()))
