# Publication Readiness Report

## Generated Files

- `publication_package/manuscript.docx`
- `publication_package/manuscript.pdf`
- `publication_package/README_publication_package.md`
- `publication_package/publication_readiness_report.md`
- `publication_package/verification/qc_report.md`

## Evidence Used

- `data/research_master_dataset.csv`
- `data/dataset_v1.csv` through `data/dataset_v5.csv`
- `data/dataset_summary.csv`
- `reports/dataset_audit_report.md`
- `reports/dataset_validation_summary.csv`
- `results_standard_research_clean/`
- `results_standard_research_clean_smote/`
- `results_threshold_research_clean_no_smote/`
- `results_threshold_research_clean_smote/`
- `results_comparison_research_clean/`
- `scripts/ml_pipeline_core.py`
- `scripts/run_experiments_standard.py`
- `scripts/generate_all_versions.py`

## Figures Generated and Embedded

- Figure 1: `publication_package/figures/figure1_experimental_pipeline.png`
- Figure 2: `publication_package/figures/figure2_dataset_version_progression.png`
- Figure 3: `publication_package/figures/figure3_class_distribution.png`
- Figure 4: `publication_package/figures/figure4_fraud_f1_comparison.png`
- Figure 5: `publication_package/figures/figure5_precision_comparison.png`
- Figure 6: `publication_package/figures/figure6_recall_comparison.png`
- Figure 7: `publication_package/figures/figure7_pr_auc_comparison.png`
- Figure 8: `publication_package/figures/figure8_threshold_optimization_curve.png`
- Figure 9: `publication_package/figures/figure9_best_tuned_confusion_matrix.png`
- Figure 10: `publication_package/figures/figure10_feature_importance.png`

## Tables Generated and Embedded

- TABLE I: `publication_package/tables/table1_dataset_summary.csv`
- TABLE II: `publication_package/tables/table2_progressive_dataset_versions.csv`
- TABLE III: `publication_package/tables/table3_experimental_configuration.csv`
- TABLE IV: `publication_package/tables/table4_best_default_no_smote.csv`
- TABLE V: `publication_package/tables/table5_best_default_smote.csv`
- TABLE VI: `publication_package/tables/table6_threshold_tuned_comparison.csv`
- TABLE VII: `publication_package/tables/table7_summary_of_main_findings.csv`

## References Added

- IEEE reference list entries: 30
- `publication_package/references/references.bib`
- `publication_package/references/references_ieee.txt`

## Verification Result

- QC status: PASS
- Repeated figure/table embedding verification: 20/20 PASS
- PDF page render count: 16

## What Remains Weak

- The dataset is synthetic and lacks an externally documented institutional source.
- There is no external validation dataset.
- MerchantRisk and CardRisk remain leakage-risk proxy features.
- Confidence intervals, repeated-seed robustness checks, and formal significance testing are not present.
- The document is not yet converted into an official IEEE template submission format.

## What Still Needs Human Review

- Author names, affiliations, acknowledgments, and funding statement.
- Venue-specific IEEE formatting requirements.
- Reference formatting and bibliographic completeness.
- Ethical/data availability statement required by the target venue.
- Whether Figure 10 impurity-based importance is sufficient or should be replaced by a more robust interpretability analysis.

## Readiness Verdict

**TECHNICALLY COMPLETE FOR AUTHOR REVIEW; NOT FINAL IEEE SUBMISSION UNTIL HUMAN METADATA AND VENUE FORMATTING ARE ADDED.**
