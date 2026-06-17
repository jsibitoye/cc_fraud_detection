# UI Report Readiness Report

**Document:** UI_Fraud_Detection_Final_Report.docx / .pdf
**Title:** Progressive Feature Engineering and Imbalanced Machine Learning for Credit Card Fraud Detection
**Student:** Joshua Ibitoye (Matric No: 225791), M.Sc. Information Security, University of Ibadan
**Date:** 17 June 2026

## What was generated

A full University of Ibadan-style M.Sc. dissertation produced as both a Word document and a PDF, formatted to the University of Ibadan Postgraduate College guidelines and the University of Ibadan Manual of Style (2023). The document is 50 pages and comprises: complete front matter (flyleaf placeholder, title page, certification, dedication, acknowledgements, abstract with keywords, list of abbreviations, an auto-measured table of contents, list of tables and list of figures); five chapters (Introduction; Literature Review; Methodology; Results and Discussion; Summary, Conclusion and Recommendations); a reference list of 32 sources; and Appendices A–E.

Two supporting deliverables were also produced: `UI_Report_QC_Checklist.md` and this readiness report.

## Evidence used

Only final, research-clean repository evidence was used:

- `data/dataset_summary.csv` and the dataset audit in `reports/dataset_audit_report.md` (dataset facts and provenance).
- `scripts/ml_pipeline_core.py` and `scripts/run_experiments_standard.py` (methodology: splits, leakage-safe feature engineering, preprocessing per model family, SMOTE placement, CV tuning on PR-AUC, selection rule).
- `results_standard_research_clean/` and `results_standard_research_clean_smote/` (default-threshold metrics).
- `results_threshold_research_clean_no_smote/` and `results_threshold_research_clean_smote/` (threshold-tuned operating points).
- `publication_package/supplementary_materials/` (overall best models, threshold best, v5 Random Forest feature importance).

Legacy material (`.old/`, `archive/`, `reports_v2/`, the raw merged file, and the simplified root `main.py`) was deliberately excluded from the methodology and results.

## Figures included (11)

Fig. 2.1 Conceptual framework; Fig. 3.1 Experimental workflow; Fig. 3.2 Progressive dataset version design; Fig. 4.1 Class distribution; Fig. 4.2 Fraud-class F1 comparison; Fig. 4.3 Precision comparison; Fig. 4.4 Recall comparison; Fig. 4.5 PR-AUC comparison; Fig. 4.6 Threshold optimisation curve; Fig. 4.7 Best tuned confusion matrix; Fig. 4.8 Feature importance (impurity-based, computed from the saved v5 Random Forest artifact and reported with an explicit interpretive caveat).

## Tables included (11)

Table 2.1 Empirical Review Comparison Matrix; Table 3.1 Dataset Summary; Table 3.2 Progressive Dataset Versions; Table 3.3 Experimental Configuration; Table 4.1 Dataset Audit Summary; Table 4.2 Best Default-Threshold Results Without SMOTE; Table 4.3 Best Default-Threshold Results With SMOTE; Table 4.4 Threshold-Tuned Comparison; Table 4.5 Summary of Findings by Research Question; Table A.1 Data Dictionary; Table C.1 Supplementary Test Metrics.

## Key reported results

- Default-threshold fraud-class F1 rises from ~0.78 (v1–v2) to ~0.84–0.86 (v3–v5).
- SMOTE improves default-threshold F1 in all versions, chiefly by raising precision while reducing recall.
- After validation-based threshold tuning, SMOTE and non-SMOTE are practically equal (tuned F1 differences < 0.001).
- Best tuned operating point: v5 No-SMOTE Random Forest at threshold 0.80 (precision ≈ 0.999, recall ≈ 0.758, fraud-class F1 ≈ 0.862).
- v3–v5 are treated as practically tied; no claim of conclusive v5 superiority is made.

## Unresolved weaknesses

- The dataset is synthetic and researcher-generated; there is no externally sourced or institutional data.
- No external validation dataset is available, so generalisation beyond the present data is not demonstrated.
- MerchantRisk and CardRisk are proxy risk indicators that retain a residual leakage risk.
- No repeated-seed robustness analysis, confidence intervals or formal significance testing are present in the pipeline.
- Feature importance is impurity-based; a model-agnostic analysis (e.g., SHAP) is recommended but not yet performed.
- No real-time, streaming or cost-sensitive deployment evaluation was undertaken.
- Front-matter metadata (Department, Faculty, Supervisor Name, Month/Year) and certification signatures remain placeholders to be completed by the student/supervisor.

## Readiness assessment

- **Ready for supervisor review:** Yes, once the administrative metadata placeholders are filled. The content, methodology, results and formatting meet M.Sc. supervisor-review expectations.
- **Ready for journal publication:** No. Publication would require, at minimum, external validation, repeated-seed robustness with confidence intervals, a leakage audit of the proxy risk features, and venue-specific formatting.

## Final verdict

**READY FOR SUPERVISOR REVIEW** (subject to completion of student/department metadata and certification signatures).
