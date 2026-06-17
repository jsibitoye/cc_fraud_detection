# Final Results Research Audit

Date: 2026-06-16

## Audit Verdict

The completed research-clean pipeline outputs are internally consistent and valid for manuscript drafting.

The results should be reported with one important nuance: SMOTE improves default-threshold Fraud F1, but after validation-based threshold tuning, SMOTE and no-SMOTE are practically tied.

## Validation Checks

- Dataset validation passed for `data/dataset_v1.csv` through `data/dataset_v5.csv`.
- Both production roots contain all five versions:
  - `results_standard_research_clean`
  - `results_standard_research_clean_smote`
- Post-training outputs passed validation:
  - `results_threshold_research_clean_no_smote`
  - `results_threshold_research_clean_smote`
  - `results_comparison_research_clean`
- Each training run records `selection_metric_source=validation` and `evaluation_metric_source=test`.
- Validation/test prediction files include `transaction_id`.
- Validation/test transaction ID overlap was checked across all versions/models and was zero.
- Calendar artifact columns (`DayOfWeek`, `Month`, `IsWeekend`, `IsWeekendDerived`) are excluded from model features.

## Default-Threshold Best Models

| Version | No-SMOTE Best | No-SMOTE Test F1 | SMOTE Best | SMOTE Test F1 | SMOTE Delta |
|---|---:|---:|---:|---:|---:|
| v1 | XGBoost | 0.777175 | DecisionTree | 0.817996 | +0.040821 |
| v2 | XGBoost | 0.778557 | RandomForest | 0.817089 | +0.038532 |
| v3 | CatBoost | 0.841437 | RandomForest | 0.861846 | +0.020409 |
| v4 | DecisionTree | 0.850685 | CatBoost | 0.861644 | +0.010959 |
| v5 | RandomForest | 0.848045 | CatBoost | 0.861735 | +0.013690 |

Default-threshold conclusion: SMOTE meaningfully improves Fraud F1 in every version, mostly by increasing precision while reducing recall modestly.

## Tuned-Threshold Best Operating Points

| Version | No-SMOTE Tuned F1 | SMOTE Tuned F1 | SMOTE Delta |
|---|---:|---:|---:|
| v1 | 0.819141 | 0.818983 | -0.000158 |
| v2 | 0.818759 | 0.818828 | +0.000070 |
| v3 | 0.861660 | 0.861611 | -0.000049 |
| v4 | 0.861826 | 0.861443 | -0.000382 |
| v5 | 0.862030 | 0.861520 | -0.000509 |

Tuned-threshold conclusion: once thresholds are selected on validation data and evaluated on test data, no-SMOTE and SMOTE are practically tied.

## Recommended Manuscript Framing

- Primary default-threshold result: SMOTE improves Fraud F1, with the best default-threshold result from v3 SMOTE RandomForest (`Fraud F1=0.861846`).
- Operating-point result: threshold tuning equalizes SMOTE and no-SMOTE, with the best tuned result from v5 no-SMOTE RandomForest (`Fraud F1=0.862030`), but differences among v3-v5 tuned models are practically negligible.
- Do not claim v5 is materially better than v3, or vice versa. Treat v3-v5 as a practical tie and discuss model/feature complexity.
- `MerchantRisk` and `CardRisk` should be described as synthetic/proxy risk indicators and included in limitations.
- The merged raw v3/v4/v5 file remains invalid as a sixth experiment and should be discussed only as provenance/appendix material.
