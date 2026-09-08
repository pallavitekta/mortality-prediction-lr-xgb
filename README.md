# Predicting Early In-Hospital Mortality from First-24-Hour Clinical Data (MIMIC-IV Demo)

## Project context

This project was developed as a hands-on application of concepts covered in **100 Days of Machine Learning**, a machine learning video series by **CampusX**. The project extends those foundational concepts into an end-to-end clinical machine learning problem using real-world electronic health record data from the MIMIC-IV Clinical Database Demo.

The goal was to practice the full workflow (data engineering, handling class imbalance, model selection, avoiding leakage, nested cross-validation, calibration, and stability analysis) on a real, messy clinical dataset. **This is not intended as a validated or deployable mortality-prediction tool**, and the performance numbers below should not be read as a claim about real-world predictive accuracy.

The dataset used here is a small, public demo subset of MIMIC-IV which has 100 patients, 275 hospital admissions, and only 15 in-hospital deaths (~5.45% event rate). With roughly 135 engineered predictors and 15 events, this is nowhere near the sample size needed for a stable, generalizable clinical model.

Because this project uses a small demonstration dataset with relatively few mortality events, the results should be considered **exploratory and educational rather than clinically deployable**.

## Ethical considerations

These were treated as first-class design decisions, not an afterthought bolted on at the end.

- **Demographic variables were deliberately excluded from modeling.**
  `race`, `insurance`, `marital_status`, and `language` were never merged
  into the feature set at all (only `admission_type` and
  `admission_location` were kept from the admissions demographic columns,
  alongside `gender`). 
  
  Reason: these variables are plausible proxies
  for structural inequity in access to and quality of care, rather than
  physiological drivers of mortality. Including them risks a model that
  learns and reproduces existing disparities under the appearance of
  objective, data-driven prediction — the same concern that has driven
  recent moves in medicine (e.g., removing race from the eGFR equation) to
  stop treating socially-patterned variables as if they were biology. No
  fairness audit was performed to test whether dropping these variables
  materially changed performance; the exclusion was made as a precautionary
  default rather than something empirically justified in this notebook.
- **Re-identification risk in a small cohort.** MIMIC-IV demo data is
  de-identified, but a 100-patient cohort has meaningfully less "safety in
  numbers" than the full MIMIC-IV database. This project makes no attempt
  to re-identify patients, does not publish patient-level row tables
  outside this notebook, and treats the data under the spirit of
  PhysioNet's Data Use Agreement even though the demo subset itself does
  not require credentialing.
- **Feature engineering was driven by clinical reasoning, not by which
  variables best predicted the outcome.** Lab tiers were
  defined by real-world ordering behavior,decided before looking at `hospital_expire_flag` at all. Selecting
  features by their raw correlation with the outcome would itself have
  been a subtle form of leakage and was avoided on principle.
- **No claim of clinical validity or regulatory approval is made or
  implied anywhere in this project.** Bodies such as the WHO do not
  approve or endorse specific algorithms (logistic regression, XGBoost, or
  otherwise). They publish algorithm-agnostic governance principles
  (transparency, lifecycle documentation, human oversight of any resulting
  decision) that any deployed clinical AI tool would need to satisfy,
  separately from whichever national regulator (FDA, MHRA, etc.) would
  need to approve a specific validated product for a specific intended
  use. This project does not attempt to meet that bar and isn't intended to.

## Data

This project uses the **MIMIC-IV Clinical Database Demo (v2.2)**, a
publicly available, de-identified subset of MIMIC-IV hosted on PhysioNet. The dataset is not included in this repository. 
* Citations - (see references.bib for full citation)

1. APA	Pollard, T., Moody, B. E., Lehman, L., Gow, B., Fernandes, C., Xie, C., Johnson, A., Mark, R. G., & Heldt, T. (2026). PhysioNet as a global platform for biomedical research. Nature Health. https://doi.org/10.1038/s44360-026-00096-z. Available from: https://rdcu.be/faatM

2. MLA	Pollard, Tom, et al. “PhysioNet as a Global Platform for Biomedical Research.” Nature Health, 2026, https://doi.org/10.1038/s44360-026-00096-z. Available from: https://rdcu.be/faatM

3. CHICAGO	Pollard, Tom, Benjamin E. Moody, Li-wei Lehman, Brian Gow, Chrystinne Fernandes, Chen Xie, Alistair Johnson, Roger G. Mark, and Thomas Heldt. “PhysioNet as a Global Platform for Biomedical Research.” Nature Health (2026). https://doi.org/10.1038/s44360-026-00096-z.i Available from: https://rdcu.be/faatM

4. HARVARD	Pollard, T., Moody, B.E., Lehman, L., Gow, B., Fernandes, C., Xie, C., Johnson, A., Mark, R.G. and Heldt, T., 2026. PhysioNet as a global platform for biomedical research. Nature Health. Available at: https://doi.org/10.1038/s44360-026-00096-z. Available from: https://rdcu.be/faatM

5. VANCOUVER	Pollard T, Moody BE, Lehman L, Gow B, Fernandes C, Xie C, et al. PhysioNet as a global platform for biomedical research. Nature Health. 2026. doi:10.1038/s44360-026-00096-z. Available from: https://rdcu.be/faatM

## Dataset access
Users should download it directly from PhysioNet and configure the local path in the notebook before running the analysis.

* Dataset: MIMIC-IV Clinical Database Demo
* Source: PhysioNet
* Cohort: 100 patients and 275 hospital admissions

1. Access page: https://physionet.org/content/mimic-iv-demo/2.2/
2. Extract the archive.
3. Set MIMIC_PATH in the notebook to the location of the extracted dataset.
4. Run the notebook from the beginning.

The MIMIC-IV dataset is de-identified and is provided for research and educational use under the terms specified by PhysioNet.

**Important**: The dataset files themselves are not redistributed with this repository. Please obtain them directly from PhysioNet.

## Problem statement

> Can clinical information available during the **first 24 hours** of hospitalization predict **in-hospital mortality** in **adult patients**?

Clinical motivation:
Early identification of patients at high risk of death could help clinicians
1. recognize high-risk patients sooner,
2. increase monitoring and reassessment, and
3. consider appropriate escalation of care.

## Notebook walkthrough (cell-wise concepts)

### 1. Data loading and admission-level structure

`admissions`/`patients`/etc. are loaded and inspected for the
`subject_id` → `hadm_id` relationship. This surfaced an important design
decision early: **100 unique patients contribute 275 admissions**, meaning
several patients are readmitted. The unit of prediction was deliberately
kept as `hadm_id` (each admission is a distinct clinical event and
readmission can represent a new problem, a relapse, or a follow-up; collapsing to one row per patient would discard that), while `subject_id`
was retained separately and used later to **group** cross-validation
splits. These are two different decisions and are not in
tension with each other.

### 2. Time-windowing (the leakage control)

Every source table (labs, vitals, microbiology, outputs, procedures,
transfers) is filtered so only events between `admittime` and
`admittime + 24h` are kept, **before** any merging into the feature set.
A model meant to predict mortality from the first 24 hours is only honest
if none of its inputs could only have been known later. This is the
single most important leakage control in the whole pipeline and was
enforced at the raw-table level.

### 3. Tiered lab feature engineering

Labs were split into three tiers by **clinical ordering behavior**, not by
correlation with the outcome (which would itself be leakage):

- **Tier 1** : routine panels ordered on essentially every admission
  (electrolytes, CBC, BMP). Only the chronologically **first** value in
  the 24-hour window is kept.
- **Tier 2** : extended electrolytes and coagulation studies, ordered
  often but not universally. Also first-value only.
- **Tier 3** : labs ordered only when a clinician already suspects
  something specific (lactate, blood gases, liver panel). Here *min*,
  *max*, and *first* are all retained, plus an explicit "was this lab
  ordered" flag because for these tests, **missingness itself reflects
  a clinical decision** and carries signal rather than being a random gap.

**Correction made during development:** the `'first'` aggregation
originally relied on whatever row order the raw CSV happened to be in,
not true chronological order. Each tier's data is now explicitly
`.sort_values('charttime')` before the `groupby(...).agg('first')` call,
so "first" genuinely means the earliest reading in the window.

### 4. Vitals and GCS

Vitals (heart rate, blood pressure, respiratory rate, SpO2, temperature)
and Glasgow Coma Scale components follow the same sort-then-aggregate
pattern as Tier-3 labs. GCS sub-scores (Eye/Motor/Verbal) are manually
mapped from MIMIC's text categories to the standard numeric scale,
including the MIMIC-specific `'No Response-ETT'` verbal category
(intubated patients) mapped to the correct clinical convention. The
category-to-score maps were checked directly against
`vitals_24h['value'].unique()` for this cohort and confirmed complete. No category present in the data falls through unmapped.

### 5. Additional predictors and demographic exclusion

Blood cultures sent, invasive procedures, ICU transfer within 24h, 24-hour
urine output, and age at admission (derived from MIMIC's shifted
`anchor_age`/`anchor_year`) are engineered the same way. Admission-level
demographics are merged in via `admission_demo_cols`, which was
deliberately narrowed to `['hadm_id', 'admission_type',
'admission_location']`.  `race`, `insurance`, `marital_status`, and
`language` are excluded at the merge step itself (see *Ethical
considerations* above), so they never enter `features` at all.

### 6. Missing data

- 100%-missing columns (e.g., estimated GFR, not captured in this cohort)
  are dropped outright.
- Near-zero-variance numeric columns were reviewed rather than
  auto-dropped, since some are electrolytes where a small spread is still
  clinically meaningful.
- Highly correlated predictor pairs (first PT vs. first INR) were checked
  and one of each redundant pair dropped. Hemoglobin and hematocrit were
  deliberately **not** treated as redundant despite their statistical
  correlation, since they measure clinically distinct properties (mass
  concentration vs. volume fraction). This was a judgment call kept in, on
  clinical grounds, rather than removed on purely statistical grounds.
- Remaining missing values are imputed with the **median** (robust to
  outlier-driven skew in lab values), inside the same `Pipeline` as the
  model and were not fit on the full dataset beforehand.
- Tier-3 "was this lab ordered" flags are retained alongside imputation,
  so missingness-as-signal isn't erased.

### 7. Categorical encoding and class imbalance

`gender`, `admission_type`, and `admission_location` are one-hot encoded
(`drop_first=True`). With only 15 deaths out of 275 admissions,
imbalance is handled via `class_weight='balanced'` (Lasso) and
`scale_pos_weight` (XGBoost), and both **ROC-AUC and PR-AUC** are tracked
throughout, since ROC-AUC alone can look deceptively strong under heavy
imbalance.

### 8. Model choice: linear vs. non-linear, and why not PCA

Two models were fit to address a genuine open question, "Is the
relationship between predictors and mortality linear in the logit, or
not?"

- **Lasso (L1) logistic regression** : the linear candidate. With ~135
  predictors and 275 admissions, L1 regularization performs simultaneous
  shrinkage and feature selection, which an unregularized logistic
  regression could not do safely at this dimensionality.
- **XGBoost** : the non-linear candidate, able to capture thresholds and
  interactions a linear-in-the-logit model can't represent, at the cost
  of higher overfitting risk with only 15 positive cases.
- **PCA was considered and rejected** as an alternative dimensionality
  reduction step: PCA is unsupervised and outcome-blind (its components
  maximize variance, with no guarantee that variance aligns with what
  separates deaths from survivors, especially for a rare outcome), and
  its components aren't directly interpretable back to named clinical
  variables the way Lasso's signed coefficients are. Lasso does
  dimensionality reduction and supervised, interpretable feature
  selection in one step, which better suited both the sample size and the
  goal of comparing feature importance against XGBoost afterward.

### 9. Patient-grouped, nested cross-validation

Evaluation went through several corrections during development, each
addressing a distinct failure mode:

- **Leakage via re-evaluating tuned hyperparameters on already-seen data**
  was avoided by moving to **nested CV**: an inner loop selects
  hyperparameters using only that fold's training data, and an outer loop
  scores the entire tuning procedure on data never touched during tuning.
- **Patient-level leakage** : since 100 patients contribute 275
  admissions, a plain row-level CV split could put two admissions from
  the same patient on both sides of a train/test boundary, letting the
  model partially learn a specific patient's signature rather than
  generalizing to new patients. Fixed by switching both the inner and
  outer splitters to **`StratifiedGroupKFold`**, grouped on `subject_id`,
  guaranteeing all of one patient's admissions stay on one side of every
  split.
- **Degenerate folds** : with only 15 total deaths, splitting further
  into 5 inner folds occasionally left a fold's *training* portion with
  zero deaths, which `LogisticRegression`/`XGBClassifier` cannot fit
  (they need both classes present to learn a boundary). This is
  distinct from a *test*-fold with zero deaths, which breaks ROC-AUC/PR-AUC
  computation but not model fitting. Resolved two ways: (1) the inner
  scoring metric was switched from `roc_auc` to **`neg_log_loss`**, which
  remains well-defined even on a single-class test fold; (2) the inner
  loop was reduced from 5 to **3 splits**, giving each fold's training
  portion a larger, more reliable share of the (few) available events. A
  `check_folds_have_events` assertion is run before every grid search to
  confirm both the training and test portion of every fold contain at
  least one event, rather than relying on the absence of a warning.

### 10. Calibration

Beyond ranking ability (ROC-AUC/PR-AUC), each model's predicted
probabilities were checked for whether they mean what they say e.g.,
whether admissions the model scored "~40% risk" actually died roughly
40% of the time. Out-of-fold predictions were collected across all 5
outer folds (`nested_oof_predictions`), giving one honestly-generated
probability per admission, from a model that never saw that admission
during training. A calibration curve (`n_bins=4`, quantile-based, given
the small event count) and the Brier score were computed for both models.
With only 15 events pooled across 4 bins, the resulting curve is visibly
noisy but informative as a diagnostic, not precise as a statistic.

### 11. Feature selection / importance stability

Rather than reporting a single full-data fit's coefficients or
importances as "the" predictors, both models were refit within each of
the 5 outer folds (`nested_feature_stability`), to check whether the same
features keep getting selected regardless of which patients happen to be
in the training set.  A feature selected in only 1 of 5 folds is a signal
the model latched onto that fold's noise, not a real pattern.

## Results

- **Nested Lasso:** ROC-AUC 0.484 ± 0.250, PR-AUC 0.165 ± 0.150 : 
  essentially **chance-level discrimination** once patient-level leakage
  was corrected. Earlier, pre-fix iterations of this notebook showed a
  visibly better Lasso score; that improvement did not survive proper
  grouped, nested evaluation, and is presented here as a methodological
  finding, not a disappointing result to downplay.
- **Nested XGBoost:** ROC-AUC 0.696 ± 0.193, PR-AUC 0.227 ± 0.128 — a
  more encouraging point estimate, but the confidence interval is wide.
- **Lasso and XGBoost's confidence intervals overlap substantially**
  (Lasso ROC-AUC roughly 0.23–0.73; XGBoost roughly 0.50–0.89). This
  dataset cannot currently support a confident claim that XGBoost
  outperforms Lasso, only that its point estimate is higher.
- **Feature stability (Lasso):** of the full-data model's ~35 non-zero
  coefficients, only a handful - `ordered_pH`, `first_Potassium`,
  `age_at_admission`, and `total_urine_output_24h_ml` were selected in
  **all 5** outer folds. Most other features were selected in 3 or fewer
  folds, consistent with the sample-size-driven instability expected at
  this event count.
- **Feature stability (XGBoost):** `min_Lactate` had both the highest
  mean importance and appeared in the top-10 most-important features in
  4 of 5 outer folds, a reassuring point of partial agreement with
  clinical intuition (lactate is a well-established severity marker),
  though still not stable across every fold.

## Limitations

- **Sample size.** 15 events is too few for stable coefficient estimates,
  stable feature selection, or a tight confidence interval on any reported
  AUC. Treat reported scores as noisy, wide-interval estimates, not
  precise performance claims.
- **Demo cohort, not the full MIMIC-IV database.** Findings here are not
  claimed to generalize beyond this specific 100-patient demo subset.
- **No external validation cohort.** Performance was assessed only via
  internal nested cross-validation on the same 275 admissions.
- **Feature selection instability**, quantified directly in the Results
  section above rather than only asserted qualitatively.
- **Calibration is descriptive, not confirmatory**, given only 15 events
  spread across a 4-bin curve.
- **No fairness audit was performed** to empirically test whether
  excluding `race`/`insurance`/`marital_status`/`language` changed model
  performance. The exclusion was a precautionary ethical decision, not
  a data-driven one (see *Ethical considerations*).
- **Planned next step:** re-running this same pipeline against a larger
  subset of the full, credentialed MIMIC-IV database (likely via
  PhysioNet's BigQuery access, given the size of the full tables) to
  check whether these findings (particularly Lasso's collapse to
  chance-level performance)hold, improve, or change at a realistic
  event count.

## Repository contents

- Full notebook : `mortality_prediction_XGB.ipynb`.
