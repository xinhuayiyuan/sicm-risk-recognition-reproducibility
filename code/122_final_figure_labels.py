# 120_final_figure_labels.py — analysis-set labels for supplementary forest and SHAP figures
# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

D = Path('<PROJECT_ROOT>/exec')
# Subgroup forest: every row carries its own denominator and events.
df = pd.read_csv(D / 'results/v5_subgroup_heterogeneity.csv').dropna(subset=['AUC'])
labels = []
for r in df.itertuples():
    subgroup = r.subgroup
    label = subgroup if subgroup.startswith(str(r.dim)) else f"{r.dim}: {subgroup}"
    labels.append(f"{label} (n={int(r.n):,}; {int(r.events):,} SICM)")
y = np.arange(len(df))[::-1]
fig, ax = plt.subplots(figsize=(10, 6.5))
ax.errorbar(df.AUC, y, xerr=[df.AUC-df.lo, df.hi-df.AUC], fmt='o', color='#1f77b4',
            ecolor='#1f77b4', capsize=3, ms=5)
ax.axvline(0.738, color='#d62728', ls='--', lw=1.1, label='Overall OOF AUC = 0.738')
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8.5)
ax.set_xlim(0.4, 1.0); ax.set_xlabel('Out-of-fold AUC (95% CI)')
ax.set_title('ICD-10-only HF-ascertainment cohort (n = 1,504; 676 SICM)\nPrespecified subgroup robustness analysis; strict 0–24 h predictors', fontsize=11)
ax.legend(loc='lower right', fontsize=8.5); ax.grid(alpha=0.25, axis='x')
fig.tight_layout(); fig.savefig(D / 'figures/figS_subgroup_forest.png', dpi=300); plt.close(fig)

# Add provenance without altering any SHAP coordinate, colour or feature value.
src = D / 'figures/fig4_shap_corrected.png'
out = D / 'figures/fig4_shap_cohort_labelled.png'
im = Image.open(src).convert('RGB')
band = max(120, int(im.width * 0.045))
canvas = Image.new('RGB', (im.width, im.height + band), 'white')
canvas.paste(im, (0, band)); draw = ImageDraw.Draw(canvas)
font_path = Path(matplotlib.get_data_path()) / 'fonts/ttf/DejaVuSans.ttf'
font = ImageFont.truetype(str(font_path), max(22, int(im.width * 0.012)))
lines = ['MIMIC new-onset cohort: n = 789 (215 SICM; 574 reference)',
         'Exploratory XGBoost fitted in the full cohort | Strict 0–24 h predictors | SHAP explanation']
for i, text in enumerate(lines):
    bbox = draw.textbbox((0, 0), text, font=font)
    draw.text(((im.width-(bbox[2]-bbox[0]))/2, 10+i*(band-15)/2), text, fill='black', font=font)
canvas.save(out, dpi=(300, 300))
print('saved:', D / 'figures/figS_subgroup_forest.png')
print('saved:', out)
