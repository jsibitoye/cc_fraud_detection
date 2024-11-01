import os
import time
import joblib
import argparse
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

# Stronger models
try:
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier
    EXTRA_MODELS_AVAILABLE = True
except ImportError:
    print("[WARN] XGBoost, LightGBM, or CatBoost not installed. Run: pip install xgboost lightgbm catboost")
    EXTRA_MODELS_AVAILABLE = False


# ===============================
# Load Data
# ===============================
import csv

def load_data(file_path: str) -> pd.DataFrame:
    with open(file_path, "r", encoding="utf-8-sig") as f:
        sample = f.read(2048)
        dialect = csv.Sniffer().sniff(sample)
        delimiter = dialect.delimiter

    df = pd.read_csv(file_path, delimiter=delimiter, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    required_cols = [
        "Amount", "Merchant", "Category", "CardType", "Location", "Time",
        "FraudFlag", "HighAmountFlag", "IsNightTransaction", "MerchantRisk", "CardRisk"
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print("\n[DEBUG] Columns loaded:", df.columns.tolist())
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df["Time"] = df["Time"].astype(str)

    print("\n📌 Loaded columns:", df.columns.tolist())
    return df


# ===============================
# Preprocess Data
# ===============================
def preprocess_data(df: pd.DataFrame):
    X = df.drop("FraudFlag", axis=1)
    y = df["FraudFlag"]

    categorical_cols = ["Merchant", "Category", "CardType", "Location", "Time"]
    numeric_cols = ["Amount"]
    passthrough_cols = ["HighAmountFlag", "IsNightTransaction", "MerchantRisk", "CardRisk"]

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[("encoder", OneHotEncoder(handle_unknown="ignore"))])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
            ("pass", "passthrough", passthrough_cols),
        ]
    )
    return X, y, preprocessor


# ===============================
# Train and Compare Models
# ===============================
def train_and_compare(X, y, preprocessor, results_dir):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Preprocessing
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Balance classes with SMOTE
    smote = SMOTE(
        random_state=42,
        k_neighbors=min(5, sum(y_train == 1) - 1) if sum(y_train == 1) > 1 else 1
    )
    X_train_sm, y_train_sm = smote.fit_resample(X_train_proc, y_train)

    imbalance_ratio = (sum(y_train == 0) / sum(y_train == 1)) if sum(y_train == 1) > 0 else 1

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000, class_weight="balanced", solver="lbfgs", random_state=42
        ),
        "DecisionTree": DecisionTreeClassifier(
            class_weight="balanced", max_depth=10, random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=12, class_weight="balanced",
            random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42
        ),
    }

    if EXTRA_MODELS_AVAILABLE:
        models.update({
            "XGBoost": XGBClassifier(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                use_label_encoder=False,
                scale_pos_weight=imbalance_ratio,
                random_state=42,
                n_jobs=-1
            ),
            "LightGBM": LGBMClassifier(
                n_estimators=400,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=imbalance_ratio,
                random_state=42,
                n_jobs=-1
            ),
            "CatBoost": CatBoostClassifier(
                iterations=400,
                learning_rate=0.05,
                depth=6,
                eval_metric="AUC",
                scale_pos_weight=imbalance_ratio,
                verbose=False,
                random_state=42
            )
        })

    best_model, best_name, best_auc = None, "", -1
    best_preprocessor, best_test_report, best_cm, best_pr_auc, best_fraud_metrics = None, "", None, -1, None

    os.makedirs(results_dir, exist_ok=True)
    metrics_path = os.path.join(results_dir, "metrics.txt")

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("====================== Model Evaluation ======================\n\n")

        for name, model in models.items():
            print(f"\n[INFO] Training {name} ...")
            start = time.time()
            model.fit(X_train_sm, y_train_sm)
            elapsed = time.time() - start
            print(f"[DONE] {name} in {elapsed:.2f} sec")

            # Predictions
            y_test_pred = model.predict(X_test_proc)
            y_test_proba = model.predict_proba(X_test_proc)[:, 1]

            # Metrics
            test_report = classification_report(y_test, y_test_pred, zero_division=0)
            cm = confusion_matrix(y_test, y_test_pred)
            roc_auc = roc_auc_score(y_test, y_test_proba)
            pr_auc = average_precision_score(y_test, y_test_proba)

            fraud_metrics = classification_report(
                y_test, y_test_pred, output_dict=True, zero_division=0
            )["1"]
            fraud_precision = np.round(fraud_metrics["precision"], 4)
            fraud_recall = np.round(fraud_metrics["recall"], 4)
            fraud_f1 = np.round(fraud_metrics["f1-score"], 4)

            # Save per-model metrics
            f.write(f"=== {name} ===\n")
            f.write(test_report + "\n")
            f.write(f"Confusion Matrix:\n{cm}\n")
            f.write(f"ROC-AUC: {roc_auc:.4f}\n")
            f.write(f"PR-AUC: {pr_auc:.4f}\n")
            f.write(f"Fraud Metrics: Precision={fraud_precision}, Recall={fraud_recall}, F1={fraud_f1}\n")
            f.write(f"Training time: {elapsed:.2f} sec\n")
            f.write("=" * 60 + "\n\n")

            if roc_auc > best_auc:
                best_model = model
                best_name = name
                best_auc = roc_auc
                best_preprocessor = preprocessor
                best_test_report = test_report
                best_cm = cm
                best_pr_auc = pr_auc
                best_fraud_metrics = (fraud_precision, fraud_recall, fraud_f1)

    return best_model, best_name, best_auc, best_test_report, best_cm, best_pr_auc, best_fraud_metrics, best_preprocessor


# ===============================
# Save Best Model
# ===============================
def save_best_model(model, preprocessor, name, auc, test_report, cm, pr_auc, fraud_metrics, models_dir, results_dir):
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])
    model_path = os.path.join(models_dir, f"{name.lower()}_model.joblib")
    joblib.dump(pipeline, model_path)

    report_path = os.path.join(results_dir, "best_model_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Best Model: {name}\n\n")
        f.write("Test Report:\n")
        f.write(test_report + "\n")
        f.write(f"Confusion Matrix:\n{cm}\n")
        f.write(f"ROC-AUC: {auc:.4f}\n")
        f.write(f"PR-AUC: {pr_auc:.4f}\n")
        f.write("Fraud Metrics:\n")
        f.write(f"Precision: {fraud_metrics[0]}, Recall: {fraud_metrics[1]}, F1: {fraud_metrics[2]}\n")

    print(f"[RESULT] Best model: {name} (ROC-AUC={auc:.4f}, PR-AUC={pr_auc:.4f})")
    print(f"[SAVED] Model saved to: {model_path}")
    print(f"[SAVED] Report saved to: {report_path}")


# ===============================
# Main
# ===============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to dataset CSV")
    parser.add_argument("--version", type=str, required=True, help="Dataset version label (e.g., v1, v2)")
    args = parser.parse_args()

    results_dir = f"results_{args.version}"
    models_dir = f"models_{args.version}"

    df = load_data(args.data)
    print("\n🔎 Class distribution:")
    print(df["FraudFlag"].value_counts())

    X, y, preprocessor = preprocess_data(df)
    best_model, best_name, best_auc, best_test_report, best_cm, best_pr_auc, best_fraud_metrics, best_preprocessor = train_and_compare(X, y, preprocessor, results_dir)
    save_best_model(best_model, best_preprocessor, best_name, best_auc, best_test_report, best_cm, best_pr_auc, best_fraud_metrics, models_dir, results_dir)


if __name__ == "__main__":
    main()
