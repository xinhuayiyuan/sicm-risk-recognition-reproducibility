# Round-6 post-hoc sensitivity analyses

These analyses were added after methodological review on 2026-11-02. They are not part of the original Round-4 data lock and were not used to select an HF scheme by statistical significance or AUC.

## Reproduction order

Run from the project root after the original cohort and strict-24-h datasets have been built:

1. `python code/round6_prepare.py` (requires `MIMIC_PASSWORD` in the environment)
2. `python code/round6_models.py`
3. `python code/round6_time_mortality.py`
4. `Rscript code/round6_time_mortality.R`
5. `python code/round6_render.py`

The preparation script classifies HF using earlier versus index admissions, derives complete SICM onset from suspected infection, imaging and troponin timestamps, and reproduces the locked measured-selection weights. The model script evaluates three HF classifications and patient-grouped validation. The R survival models use patient-clustered robust variance.

Only aggregate CSV results are included. Record-level timing, start-stop and landmark files remain local and are not redistributed under the PhysioNet data-use agreements.
