# Appendix C: Reproducibility Notes

- Active datasets: `data/research_master_dataset.csv` and `data/dataset_v1.csv` through `data/dataset_v5.csv`.
- Active scripts: `scripts/generate_all_versions.py`, `scripts/ml_pipeline_core.py`, and `scripts/run_experiments_standard.py`.
- Active result folders: `results_standard_research_clean/`, `results_standard_research_clean_smote/`, `results_threshold_research_clean_no_smote/`, `results_threshold_research_clean_smote/`, and `results_comparison_research_clean/`.
- SMOTE was applied only inside the training pipeline after preprocessing and inside cross-validation folds.
- Final reporting used the untouched test split.
