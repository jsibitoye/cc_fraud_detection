# Appendix A: Dataset Audit Summary

# Dataset Audit Report

## Source Files Inspected

- `data_old\nigeria_credit_card_fraud_dataset_v1.csv`: missing
- `data_old\nigeria_credit_card_fraud_dataset_v2.csv`: missing
- `data_old\nigeria_credit_card_fraud_dataset_v3.csv`: missing
- `data_old\nigeria_credit_card_fraud_dataset_v4.csv`: missing
- `data_old\nigeria_credit_card_fraud_dataset_v5.csv`: missing
- `data\nigeria_credit_card_fraud_dataset_v1.csv`: present
- `data\nigeria_credit_card_fraud_dataset_v2.csv`: present
- `data\nigeria_credit_card_fraud_dataset_v3.csv`: present
- `data\nigeria_credit_card_fraud_dataset_v4.csv`: present
- `data\nigeria_credit_card_fraud_dataset_v5.csv`: present
- `data\nigeria_credit_card_merged_v3_v4_v5.csv`: present
- `data\nigeria_credit_card_master_clean.csv`: present
- `data\nigeria_credit_card_master_enhanced.csv`: present
- `data\dataset_v1.csv`: present
- `data\dataset_v2.csv`: present
- `data\dataset_v3.csv`: present
- `data\dataset_v4.csv`: present
- `data\dataset_v5.csv`: present

## Current Generation Lineage

- `data/genrate_all_versions.py` is a legacy typo-named generator that resamples data and can upsample with jitter.
- `scripts/generate_v5.py` creates a size/ratio-controlled v5 file by resampling and can duplicate rows when upsampling.
- `data/clean_merged_master.py` cleaned `nigeria_credit_card_merged_v3_v4_v5.csv` and removed duplicate rows/transaction identities.
- `data/enhance_master_dataset.py` added daily-life category/subcategory enrichment.
- `data/create_stratified_versions.py` created previous `dataset_v1.csv` through `dataset_v5.csv` as increasing-size stratified samples from the enhanced master.
- New `scripts/generate_all_versions.py` rebuilds all final training datasets from one unique master population with deterministic transaction IDs.

## Duplicate And Master-Dataset Findings

- Merged input rows: 849999
- Exact duplicate rows removed from merged input: 350014
- Duplicate transaction identities detected in merged input: 350014
- Additional duplicate synthetic transaction IDs removed after exact dedupe: 0
- Final clean master rows: 499985
- Near-duplicate records retained for review: 0
- Near-duplicate groups retained for review: 0

## Merged File Validity

`data/nigeria_credit_card_merged_v3_v4_v5.csv` is invalid as a sixth main training experiment. It is a stacked v3/v4/v5-style file with repeated transaction identities. It is retained only for provenance and appendix-use audit.

### v3/v4/v5 Source Overlap

- Original v3 unique IDs: 250000
- Original v4 unique IDs: 249986
- Original v5 unique IDs: 193547
- v3-v4 overlap: 0
- v3-v5 overlap: 0
- v4-v5 overlap: 193547

## Final Dataset Versions

- `v1`: 499985 rows, 8 columns, fraud ratio 0.125004; features: TransactionID|Amount|Merchant|Category|CardType|Location|Time|FraudFlag
- `v2`: 499985 rows, 11 columns, fraud ratio 0.125004; features: TransactionID|Amount|Merchant|Category|CardType|Location|Time|FraudFlag|HighAmountFlag|IsNightTransaction|SubCategory
- `v3`: 499985 rows, 13 columns, fraud ratio 0.125004; features: TransactionID|Amount|Merchant|Category|CardType|Location|Time|FraudFlag|HighAmountFlag|IsNightTransaction|SubCategory|MerchantRisk|CardRisk
- `v4`: 499985 rows, 21 columns, fraud ratio 0.125004; features: TransactionID|Amount|Merchant|Category|CardType|Location|Time|FraudFlag|HighAmountFlag|IsNightTransaction|SubCategory|MerchantRisk|CardRisk|Hour|DayOfWeek|Month|IsWeekend|AmountLog|AmountToCategoryMedian|AmountZScoreByCategory|AmountPercentile
- `v5`: 499985 rows, 23 columns, fraud ratio 0.125004; features: TransactionID|Amount|Merchant|Category|CardType|Location|Time|FraudFlag|HighAmountFlag|IsNightTransaction|SubCategory|MerchantRisk|CardRisk|Hour|DayOfWeek|Month|IsWeekend|AmountLog|AmountToCategoryMedian|AmountZScoreByCategory|AmountPercentile|NightHighAmountFlag|RiskScoreComposite

## Data Quality Notes

- All final datasets are built from the same unique clean master population to avoid row-count confounding.
- `TransactionID` is deterministic and derived from stable transaction fields using SHA-256 hashing.
- Exact duplicate rows and duplicate transaction IDs were removed from the master.
- Near-duplicates were reported but not silently deleted.
- Time-only values are parsed against a fixed date (`2024-01-01`) to avoid run-date-dependent timestamps.
- Categorical text fields are stripped and normalized; blank/unknown categorical values are counted in the CSV audit summary.
- Legacy `MerchantRisk` and `CardRisk` are retained only in v3+ as documented synthetic risk indicators; they should be discussed as leakage-risk features in reporting.

## Final Recommendation

Use `data/dataset_v1.csv` through `data/dataset_v5.csv` for main training. Do not use `data/nigeria_credit_card_merged_v3_v4_v5.csv` as a separate main experiment. Keep the merged file for appendix/provenance only.

Archived replaced dataset outputs at: `data\archive\dataset_rebuild_20260613_011112`
