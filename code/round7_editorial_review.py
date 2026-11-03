# -*- coding: utf-8 -*-
"""Editorial and statistical review without changing locked estimates."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PKG=ROOT/"投稿包_BMC"
def replace_file(path,edits):
    text=path.read_text(encoding="utf-8")
    for old,new in edits:
        if old in text: text=text.replace(old,new)
    path.write_text(text,encoding="utf-8")

main_edits=[
("Three findings emerge from the corrected analysis. First, operational SICM remained associated with higher 28-day mortality after covariate adjustment and measured-selection weighting. Second, the association was sensitive to HF classification, although it persisted under strict-time exclusion and exposure-time-sensitive models. Third, risk-recognition performance was limited and calibration unstable, particularly under patient-grouped evaluation. These findings support an association within a selected operational cohort, not a clinically deployable prediction rule.",
 "Operational SICM remained associated with higher 28-day mortality after covariate adjustment and measured-selection weighting. The association persisted under strict-time exclusion and exposure-time-sensitive models but varied with HF classification. In contrast, discrimination was limited and calibration unstable, particularly under patient-grouped evaluation. These results do not support a clinically deployable prediction rule."),
("These are not cosmetic distinctions; they alter the estimand and the cases being modelled.",
 "These changes altered both the estimand and the case mix."),
("This distinction is central to interpreting the work and to avoiding retrospective claims of prespecification.",
 "The corrected estimates therefore represent a new post-hoc analysis, not a sensitivity analysis on an unchanged cohort."),
("No test-set imputation, variable selection or model selection was performed.",
 "All preprocessing estimates were obtained from development data only."),
("The current-specification, patient-clustered full-procedure IPW bootstrap yielded an interval of 0.84–4.54, demonstrating that the weighted estimate remained uncertain despite the positive point estimate.",
 "The current-specification, patient-clustered full-procedure IPW bootstrap yielded an interval of 0.84–4.54. This interval includes the null and limits the strength of inference from the weighted point estimate."),
("The appropriate conclusion is that no positive indirect association was detected, not that vasopressors play no mediating role.",
 "Accordingly, the results indicate no detected positive indirect association; they do not exclude mediation."),
("The immediate implication is methodological: imaging and coding definitions must be resolved before model development.",
 "Future work should resolve imaging and coding definitions before model development."),
("The current model, integer score and test-set threshold are exploratory and unsuitable for clinical triage.",
 "The current model, integer score and test-set threshold remain exploratory and should not be used for clinical triage."),
]
replace_file(PKG/"Manuscript_SICM_BMC.md",main_edits)
print("Main editorial review completed")
supp_path=PKG/"Supplement_SICM_BMC.md"
text=supp_path.read_text(encoding="utf-8")
text=text.replace("for: Corrected Round-7 post-hoc analysis","for: Left Ventricular Systolic Dysfunction in Sepsis without Index-Admission Heart Failure Diagnosis")
text=text.replace("| nan | 613 | 111 | 1.782 (1.152?2.757) | 0.0095 | 0.632 |","| None (analysed cohort) | 613 | 111 | 1.782 (1.152–2.757) | 0.0095 | 0.632 |")
text=text.replace("Restrictions use the corrected cohort. `first_icu_stay` is a MIMIC-derived flag and is not a substitute for patient-grouped validation.",
 "Restrictions use the corrected cohort. Test AUCs are single-split estimates and are not independent validations. The MIMIC-derived `first_icu_stay` flag is not a substitute for patient-grouped validation or for a first-ever-stay analysis.")
text=text.replace("no pre-ICU start was included. AKI used KDIGO records in the same window.",
 "no pre-ICU start was included. Thus, pre-existing infusions are not counted as new ICU starts, and these flags should not be interpreted as complete capture of ongoing pre-ICU treatment. AKI used KDIGO records in the same window.")
text=text.replace("Propensity AUC was 0.713; truncated weights ranged from 0.20 to 3.40 and effective sample size was 876. IPW corrects measured selection only.",
 "Propensity AUC was 0.713; truncated weights ranged from 0.20 to 3.40 and effective sample size was 876. Propensity covariates were median-imputed in the source population, and the propensity model was not selected using mortality. IPW corrects measured selection only; it does not address unmeasured clinical judgement or guarantee positivity.")
text=text.replace("Pure-rounding test AUC was 0.618 (0.510–0.722); forced-sign AUC was 0.644.",
 "The unpenalised refit and integerisation are post-selection procedures, not an inferential refit or an independently validated model. Pure-rounding test AUC was 0.618 (0.510–0.722); forced-sign AUC was 0.644.")
text=text.replace("These HRs use age, sex and Charlson index as the reduced baseline adjustment set and are not numerically equivalent to the fixed-exposure OR.",
 "These HRs use age, sex and Charlson index as the reduced baseline adjustment set and are not numerically equivalent to the fixed-exposure OR. AUC intervals resample the fixed OOF predictions and therefore do not include all refitting and partition variability.")
text=text.replace("The 28-day Aalen–Johansen estimate was 17.9% (naive 1−KM 19.3%).",
 "The 28-day Aalen–Johansen estimate was 17.9% (naive 1−KM 19.3%). Fine–Gray models included lactate, SOFA and age only and were not fully adjusted for the mortality covariate set.")
text=text.replace("These analyses address coding-era classification, not complete HF timing or clinical de novo status.",
 "These analyses address coding-era classification, not complete HF timing or clinical de novo status. The legacy adjusted OR was 1.15 (P=0.333), and legacy IPW OR was 1.12 (P=0.465); the corresponding current estimates were 1.78 (P=0.0097) and 1.94 (P=0.0145).")
anchor="Contents: S1 Eligibility"
lock_note=("Analysis version and data lock. Round-7 is a post-hoc corrected analysis. The derived inputs are frozen in `ROUND7_DATA_LOCK_STATEMENT.md` with SHA256 hashes and read-only permissions. No result was changed after lock. AUC estimates from the robustness battery in Supplementary Tables S5–S6 use the separate 1,018-stay legacy ICD-10-only cohort and must not be substituted for the current 613-stay estimates.\n\n")
if lock_note not in text: text=text.replace(anchor,lock_note+anchor,1)
anchor2="## Supplement S8. Current penalised model and integer score"
s7note=("Leave-one-era-out validation in the legacy cohort produced AUCs of 0.584 (2008–2010), 0.719 (2011–2013), 0.565 (2014–2016) and 0.546 (2017–2019). The 2020–2022 block was not estimable because of sparse events. These are temporal sensitivity analyses, not independent temporal external validation.\n\n")
if s7note not in text: text=text.replace(anchor2,s7note+anchor2,1)
supp_path.write_text(text,encoding="utf-8")
tables_path=PKG/"Tables_BMC.md"
tables_text=tables_path.read_text(encoding="utf-8").replace("for: Corrected Round-7 analysis","for: Left Ventricular Systolic Dysfunction in Sepsis without Index-Admission Heart Failure Diagnosis")
tables_path.write_text(tables_text,encoding="utf-8")
print("Supplement statistical review completed")
