# Final Claims Audit

Date: 2026-06-17

| Claim | Classification | Evidence / Action |
|---|---|---|
| Dataset size and class distribution | Strongly supported | Supported by data/dataset_summary.csv and validation summary. |
| No missing values and no duplicates | Strongly supported | Supported by dataset validation and audit reports. |
| Five-version progressive feature engineering design | Strongly supported | Supported by dataset_v1.csv through dataset_v5.csv and generator lineage. |
| SMOTE improves default-threshold F1 | Strongly supported | Supported for all five versions in results_comparison_research_clean. |
| Threshold tuning removes most practical SMOTE advantage | Supported with limitation | Supported by threshold F1 values differing by less than 0.001 among strongest settings. |
| Tree-based models dominate | Supported with limitation | Best selected models are tree/boosting families, but no repeated-seed uncertainty exists. |
| v3-v5 are practically close | Supported with limitation | Supported numerically; formal statistical testing is absent. |
| Dataset is synthetic | Strongly supported | Documented in dataset audit and manuscript scope. |
| No external validation | Strongly supported | No external validation dataset is present in repository evidence. |
| MerchantRisk/CardRisk leakage risk | Supported with limitation | Feature provenance requires caution; not proven unsafe but risk is real. |
| Best tuned F1 result | Strongly supported | v5 No-SMOTE RandomForest threshold 0.80 F1 0.862030. |
| No language-processing or text-source methodology | Strongly supported | Repository features are tabular transaction fields. |
| No live decisioning evaluation | Strongly supported | No live serving or field-evaluation workflow is present. |

## Removed or Softened Claims

- Removed broad wording that could imply external generalization.
- Softened v5 interpretation to emphasize practical similarity among v3-v5.
- Clarified that SMOTE benefit is threshold-dependent.
- Clarified that proxy risk indicators require feature-provenance caution.
