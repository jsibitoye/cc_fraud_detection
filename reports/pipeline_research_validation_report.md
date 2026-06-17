# Pipeline Research Validation Report

Date: 2026-06-14

## Methodological Fixes Applied

- Training now uses three disjoint splits: train for model fitting/CV, validation for model-family selection, and final test for untouched performance reporting.
- Threshold tuning now chooses thresholds on validation predictions and reports the chosen operating point on test predictions.
- Prediction CSVs now include `split` and `transaction_id`, allowing validation/test overlap checks.
- Calendar artifact columns (`DayOfWeek`, `Month`, `IsWeekend`, `IsWeekendDerived`) are excluded from model features.
- Version baselines no longer receive implicit time-derived features from `Time`; only columns intentionally present in each dataset version are modeled.
- Distribution-dependent engineered features are recomputed inside the modeling pipeline from training-fold data, avoiding full-dataset quantile/median/z-score/percentile leakage.
- Stale model/result outputs were archived, not deleted, under `archive/run_outputs_20260614_171540`.

## Validation Commands Run

```powershell
$PY = "C:\Dev\cc_fraud_detection\.venv\Scripts\python.exe"
& $PY -m py_compile scripts\ml_pipeline_core.py scripts\run_experiments_standard.py scripts\tune_thresholds.py scripts\run_full_post_training_pipeline.py scripts\compare_experiment_results.py scripts\validate_datasets.py
& $PY scripts\validate_datasets.py --data_dir data --summary_out reports\dataset_validation_summary.csv
& $PY scripts\run_experiments_standard.py --data_dir data --outdir results_smoke_research_clean --versions v1 --sample_size 5000 --quick
& $PY scripts\run_experiments_standard.py --data_dir data --outdir results_smoke_research_clean_smote --versions v1 --sample_size 5000 --quick --use_smote
& $PY scripts\run_full_post_training_pipeline.py --no_smote_root results_smoke_research_clean --smote_root results_smoke_research_clean_smote --no_smote_threshold_outdir results_smoke_research_clean_threshold_no_smote --smote_threshold_outdir results_smoke_research_clean_threshold_smote --comparison_outdir results_smoke_research_clean_comparison --threshold_mode max_f1
& $PY scripts\run_experiments_standard.py --data_dir data --outdir results_multiversion_smoke_research_clean --versions v1 v2 v3 v4 v5 --sample_size 10000 --quick
& $PY scripts\run_experiments_standard.py --data_dir data --outdir results_multiversion_smoke_research_clean_smote --versions v1 v2 v3 v4 v5 --sample_size 10000 --quick --use_smote
& $PY scripts\run_full_post_training_pipeline.py --no_smote_root results_multiversion_smoke_research_clean --smote_root results_multiversion_smoke_research_clean_smote --no_smote_threshold_outdir results_multiversion_smoke_research_clean_threshold_no_smote --smote_threshold_outdir results_multiversion_smoke_research_clean_threshold_smote --comparison_outdir results_multiversion_smoke_research_clean_comparison --threshold_mode max_f1
```

## Smoke Validation Results

- Dataset validation passed for all five final datasets.
- No-SMOTE v1 smoke passed.
- SMOTE v1 smoke passed.
- Post-training v1 smoke passed.
- No-SMOTE v1-v5 smoke passed.
- SMOTE v1-v5 smoke passed.
- Post-training v1-v5 smoke passed.
- Validation/test prediction `transaction_id` overlap was checked on the v1-v5 smoke outputs and was zero.
- Model summaries record `selection_metric_source=validation` and `evaluation_metric_source=test`.

## Remaining Research Caveat

`MerchantRisk` and `CardRisk` remain synthetic risk indicators. They are not exact target encodings in the rebuilt master audit, but they should still be described as synthetic/proxy risk features in the paper.
