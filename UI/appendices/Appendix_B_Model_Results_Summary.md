# Appendix B: Model Results Summary

## No-SMOTE Results

- v1 XGBoost: precision 0.8062, recall 0.7502, fraud F1 0.7772, PR-AUC 0.8268.
- v2 XGBoost: precision 0.8094, recall 0.7500, fraud F1 0.7786, PR-AUC 0.8274.
- v3 CatBoost: precision 0.9219, recall 0.7739, fraud F1 0.8414, PR-AUC 0.8558.
- v4 DecisionTree: precision 0.9537, recall 0.7678, fraud F1 0.8507, PR-AUC 0.8337.
- v5 RandomForest: precision 0.9418, recall 0.7713, fraud F1 0.8480, PR-AUC 0.8561.

## SMOTE Results

- v1 DecisionTree: precision 0.9760, recall 0.7040, fraud F1 0.8180, PR-AUC 0.8237.
- v2 RandomForest: precision 0.9527, recall 0.7153, fraud F1 0.8171, PR-AUC 0.8265.
- v3 RandomForest: precision 0.998525, recall 0.758080, fraud F1 0.861846, PR-AUC 0.855235.
- v4 CatBoost: precision 0.9998, recall 0.7570, fraud F1 0.8616, PR-AUC 0.8564.
- v5 CatBoost: precision 0.9999, recall 0.7571, fraud F1 0.8617, PR-AUC 0.8558.

## Best Results

- Best default-threshold result: v3 SMOTE RandomForest, fraud F1 0.861846.
- Best threshold-tuned result: v5 No-SMOTE RandomForest at threshold 0.80, fraud F1 0.862030.
