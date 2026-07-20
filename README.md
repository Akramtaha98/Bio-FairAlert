<div align="center">

# Bio-FairAlert

**Responsible Early Warning in Higher Education: Multi-Attribute Fairness Auditing, Explainability Validation, and Robust Model Comparison**

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Reproducible](https://img.shields.io/badge/results-reproducible-brightgreen)
![Status](https://img.shields.io/badge/status-submission%20supplement-lightgrey)

Every number, table, and figure in the manuscript traces back to a script in this repository.

</div>

---

## About

This repository is the analysis code accompanying the manuscript above, submitted to *Computers & Education: Artificial Intelligence*. It contains the full pipeline for two case studies, **OULAD** (Open University Learning Analytics Dataset) and the **MOOC User Action Dataset**, covering feature engineering, model comparison, fairness auditing, and explainability.

## Contents

- [Quick start](#quick-start)
- [Getting the data](#getting-the-data)
- [Project structure](#project-structure)
- [Script to result mapping](#script-to-result-mapping)
- [Results archive](#results-archive)
## Quick start

```bash
# 1. Clone the repository, then create an environment
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the datasets and place them under data/ (see "Getting the data" below)

# 4. Run any script from inside src/
cd src
python repeated_cv_significance.py
python repeated_ablation.py 0   # ablation and intersectional scripts take a split index
```

All paths (dataset locations, output folder, figure folder) are defined once in `src/config.py` and can be overridden with environment variables if you prefer a different layout. Each script writes its output as JSON into `results/`, already included in this repository so results can be inspected without re-running anything.

## Getting the data

Both datasets are public and are not redistributed in this repository. Download them and place the files under `data/`, matching the layout `src/config.py` expects (or set the corresponding environment variable to point elsewhere).

| Dataset | Source | Files needed | Expected folder |
|---|---|---|---|
| OULAD | [analyse.kmi.open.ac.uk/open-dataset](https://analyse.kmi.open.ac.uk/open-dataset) | `studentInfo.csv`, `studentRegistration.csv`, `studentAssessment.csv`, `assessments.csv`, `courses.csv` | `data/oulad/` (or `OULAD_DATA_DIR`) |
| MOOC User Action Dataset | [snap.stanford.edu/data/act-mooc.html](https://snap.stanford.edu/data/act-mooc.html) | `mooc_actions.tsv`, `mooc_action_features.tsv`, `mooc_action_labels.tsv` | `data/mooc_raw/act-mooc/` (or `MOOC_RAW_DIR`) |

## Project structure

```
Bio-FairAlert/
    README.md
    LICENSE
    requirements.txt
    src/
        config.py                     central path configuration
        features.py                   OULAD feature engineering
        mi_stacking.py                OULAD feature selection and stacking ensemble
        mooc_features.py              MOOC feature engineering
        mooc_mi_stacking.py           MOOC feature selection and stacking ensemble
        repeated_cv_significance.py   Tables 1, 6 and Figures 3, 6
        analyze_repeated_cv.py
        make_plots_repeated_cv.py
        alt_meta_test.py
        tuning_experiment.py
        baseline_experiment.py
        fairness_calibration.py       Tables 2, 3, S1, S2
        fairness_calibration_imdband.py
        bootstrap_fairness.py
        fairness_missing_sensitivity.py
        make_plots_bootstrap.py
        fairness_panel_repeated.py    Table S3
        ablation_experiment.py        Table 4
        repeated_ablation.py
        intersectional_fairness.py    Table 5
        repeated_intersectional.py
        explain_oulad.py              Figures 5, 7
        explain_mooc.py
        explanation_stability.py
        make_arch_figures_v2.py       Figures 1, 2
        make_arch_figures.py          superseded, kept for the record
        make_plots.py                 shared plotting helpers
    data/                              datasets go here, not included
    results/                           JSON output and generated figures
```

## Script to result mapping

### Shared feature and model pipeline

| Script | Purpose |
|---|---|
| `features.py` | OULAD cohort construction and per-checkpoint feature engineering (behavioral, demographic, assessment). |
| `mi_stacking.py` | Mutual-information feature selection and the four-base-learner stacking ensemble (Logistic Regression, Random Forest, XGBoost, LightGBM) for OULAD. |
| `mooc_features.py`, `mooc_mi_stacking.py` | Equivalent feature engineering and stacking pipeline for the MOOC case study. |

### Model comparison (Section 4.1, 4.4; Tables 1, 6; Figures 3, 6)

| Script | Purpose |
|---|---|
| `repeated_cv_significance.py` | Repeated 3x3-fold CV of stacking vs. best individual base learner, both datasets, with Nadeau-Bengio corrected paired t-test and Holm correction. |
| `analyze_repeated_cv.py` | Aggregates repeated-CV output into reported means, SDs, and corrected p-values. |
| `make_plots_repeated_cv.py` | Generates Figures 3 and 6. |
| `alt_meta_test.py` | Alternative (LightGBM) meta-learner robustness check (Section 4.1). |
| `tuning_experiment.py` | Equal-budget hyperparameter tuning check at representative checkpoints (Section 4.1, 4.4). |
| `baseline_experiment.py` | Majority-class and Explainable Boosting Machine reference baselines. |

### Fairness calibration (Section 4.2; Tables 2, 3, S1, S2)

| Script | Purpose |
|---|---|
| `fairness_calibration.py`, `fairness_calibration_imdband.py` | Original illustrative single-split disability-targeted and imd_band-targeted equalized-odds calibration (the single-split numbers quoted for comparison). |
| `bootstrap_fairness.py` | Repeated 10-split version of both calibration runs; produces the mean and SD figures reported in Tables 2, 3, S1, S2. |
| `fairness_missing_sensitivity.py` | Sensitivity check excluding students with missing imd_band data. |
| `make_plots_bootstrap.py` | Generates Figure 4. |

### Broader fairness panel (Table S3)

| Script | Purpose |
|---|---|
| `fairness_panel_repeated.py` | TPR/FPR gap, equalized-odds difference, demographic-parity difference, and per-group expected calibration error, across 5 repeated splits, before and after calibration. |

### Protected-attribute ablation (Table 4)

| Script | Purpose |
|---|---|
| `ablation_experiment.py` | Original single-split ablation, superseded below and kept for the single-split number quoted in the text. |
| `repeated_ablation.py` | 10-repeated-split ablation that produces Table 4. |

### Intersectional fairness (Table 5)

| Script | Purpose |
|---|---|
| `intersectional_fairness.py` | Original single-split intersectional analysis, superseded below. |
| `repeated_intersectional.py` | 5-repeated-split intersectional analysis that produces Table 5. |

### Explainability (Section 4.3, 4.5; Figures 5, 7)

| Script | Purpose |
|---|---|
| `explain_oulad.py`, `explain_mooc.py` | SHAP, LIME, and rule-based explanations for both case studies. |
| `explanation_stability.py` | Perturbation-stability, deletion-test, permutation-importance, and cross-split explanation-overlap tests (Section 4.3). |

### Figures

| Script | Purpose |
|---|---|
| `make_arch_figures_v2.py` | Current figures: system architecture and intervention workflow (Figures 1, 2). |
| `make_arch_figures.py` | Earlier version of the same two figures, kept for the record and superseded by `make_arch_figures_v2.py`. |
| `make_plots.py` | Shared plotting helpers used by the scripts above. |

## Results archive

`results/` holds the raw and aggregated output of every script above, so any reported number can be traced back to a specific file without re-running the pipeline.

> `mi_stacking_partial_BUGGY.json` is an intentionally retained early, buggy run of the stacking pipeline (a since-fixed meta-learner miscalibration bug), kept to document the debugging process referenced in Section 4.1. It is not used for any reported result.

---

