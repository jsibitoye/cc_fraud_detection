from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import ParameterSampler, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Optional models
XGBOOST_AVAILABLE = False
CATBOOST_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    pass

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except Exception:
    pass


BASE_REQUIRED_COLUMNS = [
    "Amount",
    "Merchant",
    "Category",
    "CardType",
    "Location",
    "Time",
    "FraudFlag",
]

EXPECTED_COLUMNS = BASE_REQUIRED_COLUMNS + [
    "HighAmountFlag",
    "IsNightTransaction",
    "MerchantRisk",
    "CardRisk",
    "SubCategory",
]

RANDOM_STATE = 42
F1_TOLERANCE = 0.002
PR_AUC_TOLERANCE = 0.003
RECALL_TOLERANCE = 0.002

MODEL_SELECTION_RULE = (
    "Best model selection prioritizes minority-class fraud detection. "
    "Models are first filtered to those within 0.002 Fraud F1 of the best "
    "Fraud F1. Within that practical tie, models within 0.003 PR-AUC of the "
    "best remaining PR-AUC are kept. Within that tie, models within 0.002 "
    "fraud recall of the best remaining recall are kept. Remaining models "
    "are ranked by fraud precision, ROC-AUC, then lower training time. "
    "Accuracy is reported but is never used for selection."
)

CATEGORICAL_COLS = ["Merchant", "Category", "SubCategory", "CardType", "Location"]
NUMERIC_COLS = [
    "Amount",
    "HighAmountFlag",
    "IsNightTransaction",
    "MerchantRisk",
    "CardRisk",
    "Hour",
    "AmountLog",
    "AmountToCategoryMedian",
    "AmountZScoreByCategory",
    "AmountPercentile",
    "NightHighAmountFlag",
    "RiskScoreComposite",
]

EXCLUDED_MODEL_COLUMNS = [
    "FraudFlag",
    "Time",
    "TransactionID",
    # Calendar fields are excluded because the rebuilt source has strong
    # month/day artifacts that are not defensible deployment signals.
    "DayOfWeek",
    "Month",
    "IsWeekend",
    "IsWeekendDerived",
]


@dataclass
class EvalResult:
    version: str
    model_name: str
    setup: str
    training_time_sec: float
    accuracy: float
    fraud_precision: float
    fraud_recall: float
    fraud_f1: float
    macro_f1: float
    weighted_f1: float
    roc_auc: float
    pr_auc: float
    confusion_matrix: List[List[int]]
    report_text: str
    best_params: Dict[str, Any]


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Frequency-encodes categorical columns using relative frequencies.
    This avoids very large sparse matrices for tree-based models.
    The encoder is fit only on training data inside CV folds.
    """

    def __init__(self):
        self.frequency_maps_: Dict[str, Dict[Any, float]] = {}
        self.columns_: List[str] = []

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X).copy()
        self.columns_ = X_df.columns.tolist()
        self.frequency_maps_ = {}

        for col in self.columns_:
            freq = X_df[col].astype(str).value_counts(normalize=True, dropna=False).to_dict()
            self.frequency_maps_[col] = freq
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        out = pd.DataFrame(index=X_df.index)

        for col in self.columns_:
            fmap = self.frequency_maps_[col]
            out[col] = X_df[col].astype(str).map(fmap).fillna(0.0)

        return out.values


class LeakageSafeFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Recomputes distribution-dependent engineered features from training data.
    This prevents full-dataset quantiles, medians, z-scores, and percentiles
    from leaking held-out distribution information into validation/test folds.
    """

    def __init__(self, high_amount_quantile: float = 0.90):
        self.high_amount_quantile = high_amount_quantile

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X).copy()
        amount = pd.to_numeric(X_df["Amount"], errors="coerce") if "Amount" in X_df.columns else pd.Series(dtype=float)
        clean_amount = amount.dropna()

        if clean_amount.empty:
            self.high_amount_threshold_ = 0.0
            self.global_amount_median_ = 1.0
            self.global_amount_mean_ = 0.0
            self.global_amount_std_ = 1.0
            self.sorted_amounts_ = np.array([0.0])
        else:
            self.high_amount_threshold_ = float(clean_amount.quantile(self.high_amount_quantile))
            self.global_amount_median_ = float(clean_amount.median()) or 1.0
            self.global_amount_mean_ = float(clean_amount.mean())
            std = float(clean_amount.std())
            self.global_amount_std_ = std if std > 0 else 1.0
            self.sorted_amounts_ = np.sort(clean_amount.to_numpy(dtype=float))

        if "Category" in X_df.columns and "Amount" in X_df.columns:
            tmp = pd.DataFrame({
                "Category": X_df["Category"].astype("string").fillna("Unknown"),
                "Amount": amount,
            })
            self.category_median_ = tmp.groupby("Category")["Amount"].median().to_dict()
            self.category_mean_ = tmp.groupby("Category")["Amount"].mean().to_dict()
            self.category_std_ = tmp.groupby("Category")["Amount"].std().replace(0, np.nan).to_dict()
        else:
            self.category_median_ = {}
            self.category_mean_ = {}
            self.category_std_ = {}

        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()

        if "Amount" not in X_df.columns:
            return X_df

        amount = pd.to_numeric(X_df["Amount"], errors="coerce").fillna(self.global_amount_median_)

        if "HighAmountFlag" in X_df.columns:
            X_df["HighAmountFlag"] = (amount >= self.high_amount_threshold_).astype(int)

        if "AmountLog" in X_df.columns:
            X_df["AmountLog"] = np.log1p(amount.clip(lower=0)).round(6)

        if "Category" in X_df.columns:
            category = X_df["Category"].astype("string").fillna("Unknown")
            cat_median = category.map(self.category_median_).fillna(self.global_amount_median_).replace(0, np.nan)
            cat_mean = category.map(self.category_mean_).fillna(self.global_amount_mean_)
            cat_std = category.map(self.category_std_).fillna(self.global_amount_std_).replace(0, np.nan)

            if "AmountToCategoryMedian" in X_df.columns:
                X_df["AmountToCategoryMedian"] = (
                    amount / cat_median
                ).replace([np.inf, -np.inf], np.nan).fillna(1.0).round(6)

            if "AmountZScoreByCategory" in X_df.columns:
                X_df["AmountZScoreByCategory"] = (
                    (amount - cat_mean) / cat_std
                ).replace([np.inf, -np.inf], np.nan).fillna(0.0).round(6)

        if "AmountPercentile" in X_df.columns:
            ranks = np.searchsorted(self.sorted_amounts_, amount.to_numpy(dtype=float), side="right")
            X_df["AmountPercentile"] = (ranks / max(len(self.sorted_amounts_), 1)).round(6)

        if "NightHighAmountFlag" in X_df.columns:
            night_default = pd.Series(0, index=X_df.index)
            high_default = pd.Series((amount >= self.high_amount_threshold_).astype(int), index=X_df.index)
            night = pd.to_numeric(
                X_df.get("IsNightTransaction", night_default),
                errors="coerce",
            ).fillna(0).astype(int)
            high = pd.to_numeric(
                X_df.get("HighAmountFlag", high_default),
                errors="coerce",
            ).fillna(0).astype(int)
            X_df["NightHighAmountFlag"] = ((night == 1) & (high == 1)).astype(int)

        if "RiskScoreComposite" in X_df.columns:
            zero_default = pd.Series(0, index=X_df.index)
            high_default = pd.Series((amount >= self.high_amount_threshold_).astype(int), index=X_df.index)
            merchant_risk = pd.to_numeric(
                X_df.get("MerchantRisk", zero_default),
                errors="coerce",
            ).fillna(0).clip(0, 1)
            card_risk = pd.to_numeric(
                X_df.get("CardRisk", zero_default),
                errors="coerce",
            ).fillna(0).clip(0, 1)
            high = pd.to_numeric(
                X_df.get("HighAmountFlag", high_default),
                errors="coerce",
            ).fillna(0).astype(int)
            night = pd.to_numeric(
                X_df.get("IsNightTransaction", zero_default),
                errors="coerce",
            ).fillna(0).astype(int)
            X_df["RiskScoreComposite"] = (
                0.30 * merchant_risk
                + 0.30 * card_risk
                + 0.25 * high
                + 0.15 * night
            ).round(6)

        return X_df


class CatBoostFeatureSafeModel:
    """Applies fitted leakage-safe feature engineering before CatBoost inference."""

    def __init__(self, feature_engineer: LeakageSafeFeatureEngineer, model: Any):
        self.feature_engineer = feature_engineer
        self.model = model

    def predict(self, X):
        X_safe = _prepare_catboost_data(self.feature_engineer.transform(X))
        return self.model.predict(X_safe)

    def predict_proba(self, X):
        X_safe = _prepare_catboost_data(self.feature_engineer.transform(X))
        return self.model.predict_proba(X_safe)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_dataset(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in BASE_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{file_path.name} is missing required columns: {missing}")

    return df.copy()


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    try:
        dt = pd.to_datetime(df["Time"], errors="coerce", format="mixed")
    except TypeError:
        dt = pd.to_datetime(df["Time"], errors="coerce", infer_datetime_format=True)

    fallback = pd.Timestamp("2024-01-01 12:00:00")
    dt = dt.fillna(fallback)

    df["Hour"] = dt.dt.hour.astype(int)
    df["DayOfWeek"] = dt.dt.dayofweek.astype(int)
    df["Month"] = dt.dt.month.astype(int)
    df["IsWeekendDerived"] = (dt.dt.dayofweek >= 5).astype(int)

    return df


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    drop_cols = [c for c in EXCLUDED_MODEL_COLUMNS if c in df.columns]
    X = df.drop(columns=drop_cols).copy()
    y = df["FraudFlag"].astype(int).copy()
    return X, y


def _available_columns(feature_columns: List[str], candidates: List[str]) -> List[str]:
    return [c for c in candidates if c in feature_columns]


def build_linear_preprocessor(feature_columns: List[str]) -> ColumnTransformer:
    categorical_cols = _available_columns(feature_columns, CATEGORICAL_COLS)
    numeric_cols = _available_columns(feature_columns, NUMERIC_COLS)

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, categorical_cols),
            ("num", num_pipe, numeric_cols),
        ],
        remainder="drop",
    )


def build_tree_preprocessor(feature_columns: List[str]) -> ColumnTransformer:
    categorical_cols = _available_columns(feature_columns, CATEGORICAL_COLS)
    numeric_cols = _available_columns(feature_columns, NUMERIC_COLS)

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("freq", FrequencyEncoder()),
        ]
    )

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, categorical_cols),
            ("num", num_pipe, numeric_cols),
        ],
        remainder="drop",
    )


def make_smote_step(use_smote: bool):
    return SMOTE(random_state=RANDOM_STATE, k_neighbors=5) if use_smote else "passthrough"


def build_model_specs(
    use_smote: bool,
    imbalance_ratio: float,
    quick: bool = False,
    feature_columns: List[str] | None = None,
) -> List[Dict[str, Any]]:
    """
    Standardized search budgets across sklearn-style models.
    Search objective will be set outside this function.
    """
    specs: List[Dict[str, Any]] = []
    feature_columns = list(feature_columns or [])

    lr_c_values = [0.1, 1.0] if quick else np.logspace(-2, 2, 12)
    dt_space = {
        "clf__max_depth": [4, 8, None] if quick else [4, 6, 8, 10, 12, 16, 20, None],
        "clf__min_samples_split": [2, 20] if quick else [2, 5, 10, 20, 40],
        "clf__min_samples_leaf": [1, 8] if quick else [1, 2, 4, 8, 16],
        "clf__ccp_alpha": [0.0, 0.001] if quick else [0.0, 0.0001, 0.001, 0.01, 0.05],
        "clf__criterion": ["gini", "entropy"] if quick else ["gini", "entropy", "log_loss"],
    }
    rf_space = {
        "clf__n_estimators": [40, 80] if quick else [200, 300, 500, 700],
        "clf__max_depth": [8, None] if quick else [8, 10, 14, 18, 24, None],
        "clf__min_samples_split": [2, 10] if quick else [2, 5, 10, 20],
        "clf__min_samples_leaf": [1, 4] if quick else [1, 2, 4, 8],
        "clf__max_features": ["sqrt"] if quick else ["sqrt", "log2", 0.5, 0.8],
        "clf__bootstrap": [True] if quick else [True, False],
    }
    xgb_space = {
        "clf__n_estimators": [40, 80] if quick else [200, 300, 400, 600],
        "clf__learning_rate": [0.05, 0.1] if quick else [0.01, 0.03, 0.05, 0.1],
        "clf__max_depth": [3, 5] if quick else [3, 4, 5, 6, 8],
        "clf__subsample": [0.8] if quick else [0.7, 0.8, 0.9, 1.0],
        "clf__colsample_bytree": [0.8] if quick else [0.6, 0.8, 1.0],
        "clf__reg_lambda": [1.0, 2.0] if quick else [0.5, 1.0, 2.0, 5.0],
        "clf__min_child_weight": [1, 3] if quick else [1, 3, 5],
        "clf__gamma": [0.0] if quick else [0.0, 0.1, 0.3],
    }

    specs.append({
        "name": "LogisticRegression",
        "pipeline": ImbPipeline(steps=[
            ("feature_engineering", LeakageSafeFeatureEngineer()),
            ("preprocess", build_linear_preprocessor(feature_columns)),
            ("smote", make_smote_step(use_smote)),
            ("clf", LogisticRegression(
                class_weight="balanced",
                max_iter=5000,
                solver="lbfgs",
                random_state=RANDOM_STATE,
            )),
        ]),
        "search_space": {
            "clf__C": lr_c_values,
        },
        "n_iter": 2 if quick else 12,
    })

    specs.append({
        "name": "DecisionTree",
        "pipeline": ImbPipeline(steps=[
            ("feature_engineering", LeakageSafeFeatureEngineer()),
            ("preprocess", build_tree_preprocessor(feature_columns)),
            ("smote", make_smote_step(use_smote)),
            ("clf", DecisionTreeClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]),
        "search_space": dt_space,
        "n_iter": 2 if quick else 30,
    })

    specs.append({
        "name": "RandomForest",
        "pipeline": ImbPipeline(steps=[
            ("feature_engineering", LeakageSafeFeatureEngineer()),
            ("preprocess", build_tree_preprocessor(feature_columns)),
            ("smote", make_smote_step(use_smote)),
            ("clf", RandomForestClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
        "search_space": rf_space,
        "n_iter": 2 if quick else 30,
    })

    if XGBOOST_AVAILABLE:
        specs.append({
            "name": "XGBoost",
            "pipeline": ImbPipeline(steps=[
                ("feature_engineering", LeakageSafeFeatureEngineer()),
                ("preprocess", build_tree_preprocessor(feature_columns)),
                ("smote", make_smote_step(use_smote)),
                ("clf", XGBClassifier(
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    scale_pos_weight=1.0 if use_smote else imbalance_ratio,
                )),
            ]),
            "search_space": xgb_space,
            "n_iter": 2 if quick else 30,
        })

    return specs


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
    report_text = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Legitimate", "Fraud"],
        zero_division=0,
    )
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Legitimate", "Fraud"],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fraud_metrics = report_dict["Fraud"]

    return {
        "accuracy": round(float(report_dict["accuracy"]), 6),
        "fraud_precision": round(float(fraud_metrics["precision"]), 6),
        "fraud_recall": round(float(fraud_metrics["recall"]), 6),
        "fraud_f1": round(float(fraud_metrics["f1-score"]), 6),
        "macro_f1": round(float(report_dict["macro avg"]["f1-score"]), 6),
        "weighted_f1": round(float(report_dict["weighted avg"]["f1-score"]), 6),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 6),
        "pr_auc": round(float(average_precision_score(y_true, y_proba)), 6),
        "confusion_matrix": cm.tolist(),
        "report_text": report_text,
    }


def select_best_model(results: List[EvalResult]) -> Tuple[EvalResult, str]:
    """
    Research-grade selection rule for imbalanced fraud detection.

    The primary objective is Fraud F1. PR-AUC, recall, precision, ROC-AUC,
    and training time are used only after practical ties. Accuracy is never
    used for best-model selection.
    """
    if not results:
        raise ValueError("No model results available for selection.")

    best_f1 = max(r.fraud_f1 for r in results)
    f1_candidates = [r for r in results if r.fraud_f1 >= best_f1 - F1_TOLERANCE]

    best_pr_auc = max(r.pr_auc for r in f1_candidates)
    pr_candidates = [r for r in f1_candidates if r.pr_auc >= best_pr_auc - PR_AUC_TOLERANCE]

    best_recall = max(r.fraud_recall for r in pr_candidates)
    recall_candidates = [
        r for r in pr_candidates if r.fraud_recall >= best_recall - RECALL_TOLERANCE
    ]

    best = sorted(
        recall_candidates,
        key=lambda r: (
            r.fraud_precision,
            r.roc_auc,
            -r.training_time_sec,
        ),
        reverse=True,
    )[0]

    reason = (
        f"{best.model_name} was selected because it remained after the tolerance filters "
        f"for Fraud F1 (best={best_f1:.6f}, tolerance={F1_TOLERANCE:.3f}), "
        f"PR-AUC (best among F1-tied models={best_pr_auc:.6f}, tolerance={PR_AUC_TOLERANCE:.3f}), "
        f"and fraud recall (best among PR-AUC-tied models={best_recall:.6f}, "
        f"tolerance={RECALL_TOLERANCE:.3f}). Among the remaining models, it ranked highest "
        f"by fraud precision, then ROC-AUC, then lower training time."
    )
    return best, reason

def plot_confusion_matrix(cm: np.ndarray, model_name: str, version: str, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(2),
        yticks=np.arange(2),
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"],
        xlabel="Predicted label",
        ylabel="Actual label",
        title=f"{model_name} Confusion Matrix ({version})",
    )

    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(int(cm[i, j]), "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=11,
                fontweight="bold",
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close("all")


def plot_curves(curve_data: Dict[str, Dict[str, Any]], version: str, outdir: Path) -> None:
    plt.figure(figsize=(8, 6))
    for model_name, vals in curve_data.items():
        fpr, tpr = vals["roc_curve"]
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={vals['roc_auc']:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves ({version})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"roc_curves_{version}.png", dpi=400, bbox_inches="tight")
    plt.close("all")

    plt.figure(figsize=(8, 6))
    for model_name, vals in curve_data.items():
        precision, recall = vals["pr_curve"]
        plt.plot(recall, precision, label=f"{model_name} (PR-AUC={vals['pr_auc']:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curves ({version})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"pr_curves_{version}.png", dpi=400, bbox_inches="tight")
    plt.close("all")


def run_search_for_sklearn_model(
    spec: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
    scoring: str = "average_precision",
) -> Tuple[Any, Dict[str, Any], float]:
    """
    Standard fix:
    tune on PR-AUC / average precision instead of default-threshold F1.
    """

    search = RandomizedSearchCV(
        estimator=spec["pipeline"],
        param_distributions=spec["search_space"],
        n_iter=spec["n_iter"],
        scoring=scoring,
        n_jobs=1,
        cv=cv,
        random_state=RANDOM_STATE,
        refit=True,
        verbose=0,
    )

    start = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - start
    return search.best_estimator_, search.best_params_, elapsed


def _prepare_catboost_data(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()

    for col in _available_columns(X.columns.tolist(), CATEGORICAL_COLS):
        X[col] = X[col].fillna("Unknown").astype(str)

    for col in _available_columns(X.columns.tolist(), NUMERIC_COLS):
        med = pd.to_numeric(X[col], errors="coerce").median()
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(med)

    return X


def run_search_for_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
    use_smote: bool,
    n_iter: int = 30,
    quick: bool = False,
) -> Tuple[Any, Dict[str, Any], float]:
    """
    Standard fix:
    CatBoost tuning now uses PR-AUC / average precision on probabilities,
    not fraud-class F1 from hard predictions.
    """

    if not CATBOOST_AVAILABLE:
        raise RuntimeError("CatBoost is not available.")

    param_space = {
        "iterations": [50, 80] if quick else [200, 300, 400, 600],
        "learning_rate": [0.05, 0.1] if quick else [0.01, 0.03, 0.05, 0.1],
        "depth": [4, 6] if quick else [4, 5, 6, 8],
        "l2_leaf_reg": [3, 5] if quick else [1, 3, 5, 7, 9],
        "border_count": [64] if quick else [64, 128, 254],
    }

    if use_smote:
        pipeline = ImbPipeline(steps=[
            ("feature_engineering", LeakageSafeFeatureEngineer()),
            ("preprocess", build_tree_preprocessor(X_train.columns.tolist())),
            ("smote", make_smote_step(True)),
            ("clf", CatBoostClassifier(
                eval_metric="AUC",
                loss_function="Logloss",
                verbose=False,
                allow_writing_files=False,
                random_state=RANDOM_STATE,
            )),
        ])
        search_space = {f"clf__{key}": value for key, value in param_space.items()}
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=search_space,
            n_iter=n_iter,
            scoring="average_precision",
            n_jobs=1,
            cv=cv,
            random_state=RANDOM_STATE,
            refit=True,
            verbose=0,
        )
        start = time.time()
        search.fit(X_train, y_train)
        elapsed = time.time() - start
        return search.best_estimator_, search.best_params_, elapsed

    X_train_cb_raw = X_train.reset_index(drop=True)
    y_train_cb = y_train.reset_index(drop=True)

    sampler = list(ParameterSampler(param_space, n_iter=n_iter, random_state=RANDOM_STATE))
    best_score = -math.inf
    best_params: Dict[str, Any] | None = None

    start = time.time()

    for params in sampler:
        fold_scores: List[float] = []

        for train_idx, val_idx in cv.split(X_train_cb_raw, y_train_cb):
            X_tr_raw = X_train_cb_raw.iloc[train_idx].copy()
            X_val_raw = X_train_cb_raw.iloc[val_idx].copy()
            y_tr = y_train_cb.iloc[train_idx]
            y_val = y_train_cb.iloc[val_idx]
            fold_engineer = LeakageSafeFeatureEngineer().fit(X_tr_raw, y_tr)
            X_tr = _prepare_catboost_data(fold_engineer.transform(X_tr_raw))
            X_val = _prepare_catboost_data(fold_engineer.transform(X_val_raw))
            cat_features = _available_columns(X_tr.columns.tolist(), CATEGORICAL_COLS)

            model = CatBoostClassifier(
                **params,
                eval_metric="AUC",
                loss_function="Logloss",
                auto_class_weights="Balanced" if not use_smote else None,
                verbose=False,
                allow_writing_files=False,
                random_state=RANDOM_STATE,
            )

            model.fit(X_tr, y_tr, cat_features=cat_features)
            y_val_proba = model.predict_proba(X_val)[:, 1]
            score = average_precision_score(y_val, y_val_proba)
            fold_scores.append(float(score))

        avg_score = float(np.mean(fold_scores))
        if avg_score > best_score:
            best_score = avg_score
            best_params = params

    if best_params is None:
        raise RuntimeError("CatBoost tuning failed to select parameters.")

    final_engineer = LeakageSafeFeatureEngineer().fit(X_train_cb_raw, y_train_cb)
    X_train_cb = _prepare_catboost_data(final_engineer.transform(X_train_cb_raw))

    final_model = CatBoostClassifier(
        **best_params,
        eval_metric="AUC",
        loss_function="Logloss",
        auto_class_weights="Balanced" if not use_smote else None,
        verbose=False,
        allow_writing_files=False,
        random_state=RANDOM_STATE,
    )
    final_cat_features = _available_columns(X_train_cb.columns.tolist(), CATEGORICAL_COLS)
    final_model.fit(X_train_cb, y_train_cb, cat_features=final_cat_features)

    elapsed = time.time() - start
    return CatBoostFeatureSafeModel(final_engineer, final_model), best_params, elapsed


def save_best_model(
    best_result: EvalResult,
    best_artifact: Any,
    outdir: Path,
    selection_reason: str,
    model_names: List[str],
    dataset_path: Path,
    sample_size: int | None,
    quick: bool,
    scoring_metric: str = "average_precision",
    final_test_result: EvalResult | None = None,
    train_rows: int | None = None,
    validation_rows: int | None = None,
    test_rows: int | None = None,
    feature_columns: List[str] | None = None,
    excluded_columns: List[str] | None = None,
) -> None:
    joblib.dump(best_artifact, outdir / f"{best_result.model_name.lower()}_model.joblib")

    with open(outdir / "best_model_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Best Model: {best_result.model_name}\n\n")
        f.write("Research Setup:\n")
        f.write(f"Dataset Version: {best_result.version}\n")
        f.write(f"Dataset Path: {dataset_path}\n")
        f.write(f"Setup: {best_result.setup}\n")
        f.write(f"Quick Mode: {quick}\n")
        f.write(f"Sample Size: {sample_size if sample_size is not None else 'Full dataset'}\n")
        if train_rows is not None and validation_rows is not None and test_rows is not None:
            f.write(f"Train Rows: {train_rows}\n")
            f.write(f"Validation Rows: {validation_rows}\n")
            f.write(f"Test Rows: {test_rows}\n")
        f.write(f"Model List: {', '.join(model_names)}\n")
        f.write(f"Scoring Metric During Tuning: {scoring_metric}\n")
        f.write("Model Selection Split: validation\n")
        f.write("Final Reporting Split: test\n")
        if feature_columns is not None:
            f.write(f"Model Feature Columns: {', '.join(feature_columns)}\n")
        if excluded_columns is not None:
            f.write(f"Excluded Columns: {', '.join(excluded_columns)}\n")
        f.write("Label Convention: 0 = Legitimate, 1 = Fraud\n\n")
        f.write("Model Selection Rule:\n")
        f.write(MODEL_SELECTION_RULE + "\n\n")
        f.write("Selection Rationale:\n")
        f.write(selection_reason + "\n\n")
        f.write("Validation Metrics Used For Model Selection:\n")
        f.write(f"Accuracy: {best_result.accuracy:.6f}\n")
        f.write(f"Fraud Precision: {best_result.fraud_precision:.6f}\n")
        f.write(f"Fraud Recall: {best_result.fraud_recall:.6f}\n")
        f.write(f"Fraud F1: {best_result.fraud_f1:.6f}\n")
        f.write(f"ROC-AUC: {best_result.roc_auc:.6f}\n")
        f.write(f"PR-AUC: {best_result.pr_auc:.6f}\n")
        f.write(f"Training Time (sec): {best_result.training_time_sec:.2f}\n")
        f.write(f"Setup: {best_result.setup}\n")
        f.write(f"Best Params: {json.dumps(best_result.best_params, default=str)}\n\n")
        f.write("Classification Report:\n")
        f.write(best_result.report_text)
        f.write("\n")
        f.write(f"Confusion Matrix:\n{np.array(best_result.confusion_matrix)}\n")
        if final_test_result is not None:
            f.write("\nFinal Untouched Test Metrics:\n")
            f.write(f"Accuracy: {final_test_result.accuracy:.6f}\n")
            f.write(f"Fraud Precision: {final_test_result.fraud_precision:.6f}\n")
            f.write(f"Fraud Recall: {final_test_result.fraud_recall:.6f}\n")
            f.write(f"Fraud F1: {final_test_result.fraud_f1:.6f}\n")
            f.write(f"ROC-AUC: {final_test_result.roc_auc:.6f}\n")
            f.write(f"PR-AUC: {final_test_result.pr_auc:.6f}\n")
            f.write("Classification Report:\n")
            f.write(final_test_result.report_text)
            f.write("\n")
            f.write(f"Confusion Matrix:\n{np.array(final_test_result.confusion_matrix)}\n")
        f.write("\nLimitations:\n")
        f.write("- Hyperparameter tuning optimizes average precision using cross-validation within the training split.\n")
        f.write("- Final model-family selection uses the validation split and the documented Fraud F1-first rule.\n")
        f.write("- Final reported metrics are computed once on the untouched stratified test split.\n")
        f.write("- Threshold tuning uses validation predictions and reports the selected operating point on test predictions.\n")
