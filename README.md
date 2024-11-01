# Credit Card Fraud Detection — Minimal Pro (Nigeria-style)

This is a **clean, minimal, professional** starter for your thesis project. It uses **best practices with the least code** and is designed for **Windows 11 + VS Code**.

## 1) Setup (Windows PowerShell)
```powershell
# 1. Create & activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Upgrade pip and install deps
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2) Put data files
Create a folder named `data` in this project (already present) and copy these files into it:
- `nigeria_credit_card_fraud_dataset.csv`
- `nigeria_credit_card_fraud_dataset_2.csv`  (optional, if you want ~200k rows)

> If you downloaded from Chat, just move them into the `data/` folder.

## 3) Run the training
```powershell
# Use both files (200k rows)
python main.py --data data/nigeria_credit_card_fraud_dataset.csv data/nigeria_credit_card_fraud_dataset_2.csv

# Or use just one file (100k rows)
python main.py --data data/nigeria_credit_card_fraud_dataset.csv
```

## 4) What you get
- A trained model at `models/rf_model.joblib`
- A metrics report at `reports/metrics.txt`
- Console output with ROC-AUC, Precision/Recall/F1, and Confusion Matrix

## Notes
- We use **RandomForest (class_weight='balanced')** to handle class imbalance cleanly without SMOTE (fewer dependencies and less code).
- We extract simple time features (hour, day-of-week, is-weekend) and one-hot encode categorical features.
- Everything is wrapped in a **single scikit-learn Pipeline** to avoid leakage and keep code clean.
