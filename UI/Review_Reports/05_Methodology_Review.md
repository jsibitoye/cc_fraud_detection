# 05 Methodology Review

## Hostile finding

Issue: Chapter Three was defensible but lacked implementation-level detail on leakage-safe feature engineering, frequency encoding, random state, SMOTE parameters, threshold grid, and model-selection rules. Severity: Major before fix.

## Fix implemented

Added details from the active repository: random_state=42, SMOTE k_neighbors=5, 0.10-0.90 threshold grid, 0.01 threshold step, fold-local feature engineering, FrequencyEncoder fitting, excluded columns, and validation/test separation.

## Re-check

After improvement, the issue was re-checked in the improved manuscript. No critical or major issue remains in this review category; remaining concerns are moderate or minor and are documented as limitations.
