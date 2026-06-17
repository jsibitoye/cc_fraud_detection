from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split

from ml_pipeline_core import (
    CATBOOST_AVAILABLE,
    EvalResult,
    EXCLUDED_MODEL_COLUMNS,
    MODEL_SELECTION_RULE,
    RANDOM_STATE,
    XGBOOST_AVAILABLE,
    build_model_specs,
    ensure_dir,
    evaluate_predictions,
    load_dataset,
    plot_confusion_matrix,
    plot_curves,
    prepare_features,
    run_search_for_catboost,
    run_search_for_sklearn_model,
    save_best_model,
    select_best_model,
)

DEFAULT_VERSIONS = ["v1", "v2", "v3", "v4", "v5"]
EXPECTED_MODEL_NAMES = ["LogisticRegression", "DecisionTree", "RandomForest", "XGBoost", "CatBoost"]
PREDICTION_REQUIRED_COLUMNS = {"split", "transaction_id", "y_true", "y_pred", "y_proba"}
VALIDATION_SIZE = 0.20
TEST_SIZE = 0.20


def parse_versions(values: List[str] | None) -> List[str]:
    if not values:
        return DEFAULT_VERSIONS.copy()

    versions: List[str] = []
    for value in values:
        for part in value.split(","):
            version = part.strip().lower()
            if version:
                versions.append(version)

    invalid = [v for v in versions if v not in DEFAULT_VERSIONS]
    if invalid:
        raise ValueError(f"Unsupported version(s): {invalid}. Expected one or more of {DEFAULT_VERSIONS}.")

    return list(dict.fromkeys(versions))


def find_version_files(data_dir: Path, requested_versions: List[str] | None = None) -> List[Tuple[str, Path]]:
    candidates = []
    for version in requested_versions or DEFAULT_VERSIONS:
        options = [
            data_dir / f"dataset_{version}.csv",
            data_dir / f"nigeria_credit_card_fraud_dataset_{version}.csv",
        ]
        for p in options:
            if p.exists():
                candidates.append((version, p))
                break
        else:
            raise FileNotFoundError(
                f"No dataset file found for {version} in {data_dir}. "
                f"Checked: {[str(p) for p in options]}"
            )
    return candidates


def stratified_sample(df: pd.DataFrame, sample_size: int | None) -> pd.DataFrame:
    if sample_size is None:
        return df.copy()
    if sample_size <= 0:
        raise ValueError("--sample_size must be a positive integer.")
    if sample_size >= len(df):
        return df.copy()

    class_counts = df["FraudFlag"].value_counts()
    if len(class_counts) < 2:
        raise ValueError("Cannot stratified-sample a dataset with fewer than two classes.")
    if sample_size < len(class_counts):
        raise ValueError(
            f"--sample_size={sample_size} is too small to preserve all classes "
            f"({len(class_counts)} classes present)."
        )

    sample, _ = train_test_split(
        df,
        train_size=sample_size,
        random_state=RANDOM_STATE,
        stratify=df["FraudFlag"],
    )
    return sample.sort_index().reset_index(drop=True)


def save_prediction_csv(
    outdir: Path,
    model_name: str,
    split_name: str,
    transaction_ids: pd.Series,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> None:
    pred_df = pd.DataFrame({
        "split": split_name,
        "transaction_id": np.asarray(transaction_ids).astype(str),
        "y_true": np.asarray(y_true).astype(int),
        "y_pred": np.asarray(y_pred).reshape(-1).astype(int),
        "y_proba": np.asarray(y_proba, dtype=float).reshape(-1),
        "proba_fraud": np.asarray(y_proba, dtype=float).reshape(-1),
    })
    pred_df.to_csv(outdir / f"{model_name}_{split_name}_predictions.csv", index=False)


def validate_prediction_file(prediction_file: Path) -> None:
    if not prediction_file.exists():
        raise FileNotFoundError(f"Missing prediction file: {prediction_file}")

    df = pd.read_csv(prediction_file, nrows=5)
    missing = sorted(PREDICTION_REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"{prediction_file} is missing required column(s): {missing}")


def validate_version_outputs(version_dir: Path, version: str, best_model_name: str | None = None) -> None:
    required_files = [
        "metrics.csv",
        "metrics.txt",
        "summary.json",
        "best_model_report.txt",
        f"pr_curves_{version}.png",
        f"roc_curves_{version}.png",
    ]

    for model_name in EXPECTED_MODEL_NAMES:
        required_files.extend(
            [
                f"confusion_matrix_{model_name.lower()}_{version}.png",
                f"{model_name}_validation_predictions.csv",
                f"{model_name}_test_predictions.csv",
            ]
        )
    required_files.extend(["metrics_validation.csv", "metrics_test.csv"])

    missing = [str(version_dir / name) for name in required_files if not (version_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "[VALIDATION ERROR] Missing required output file(s):\n" + "\n".join(missing)
        )

    for model_name in EXPECTED_MODEL_NAMES:
        validate_prediction_file(version_dir / f"{model_name}_validation_predictions.csv")
        validate_prediction_file(version_dir / f"{model_name}_test_predictions.csv")

    joblib_files = sorted(version_dir.glob("*_model.joblib"))
    if not joblib_files:
        raise FileNotFoundError(f"[VALIDATION ERROR] No saved best model .joblib file found in {version_dir}")

    if best_model_name is None:
        summary_path = version_dir / "summary.json"
        if summary_path.exists():
            with open(summary_path, "r", encoding="utf-8") as f:
                best_model_name = json.load(f).get("best_model")

    if best_model_name:
        expected_best_model = version_dir / f"{best_model_name.lower()}_model.joblib"
        if not expected_best_model.exists():
            raise FileNotFoundError(
                f"[VALIDATION ERROR] Expected saved best model not found: {expected_best_model}"
            )

    print(f"[VALIDATION OK] {version_dir} contains all required training outputs.")


def evaluate_model_on_split(
    estimator: object,
    version: str,
    setup: str,
    model_name: str,
    training_time_sec: float,
    best_params: Dict[str, object],
    X_split: pd.DataFrame,
    y_split: pd.Series,
    transaction_ids: pd.Series,
    split_name: str,
    outdir: Path,
) -> tuple[EvalResult, np.ndarray, np.ndarray]:
    y_pred = np.asarray(estimator.predict(X_split)).reshape(-1).astype(int)
    y_proba = np.asarray(estimator.predict_proba(X_split)[:, 1], dtype=float).reshape(-1)

    save_prediction_csv(outdir, model_name, split_name, transaction_ids, y_split, y_pred, y_proba)
    metrics = evaluate_predictions(y_split, y_pred, y_proba)

    return (
        EvalResult(
            version=version,
            model_name=model_name,
            setup=setup,
            training_time_sec=round(training_time_sec, 2),
            accuracy=metrics["accuracy"],
            fraud_precision=metrics["fraud_precision"],
            fraud_recall=metrics["fraud_recall"],
            fraud_f1=metrics["fraud_f1"],
            macro_f1=metrics["macro_f1"],
            weighted_f1=metrics["weighted_f1"],
            roc_auc=metrics["roc_auc"],
            pr_auc=metrics["pr_auc"],
            confusion_matrix=metrics["confusion_matrix"],
            report_text=metrics["report_text"],
            best_params=best_params,
        ),
        y_pred,
        y_proba,
    )


def result_to_row(result: EvalResult, metric_split: str) -> Dict[str, object]:
    return {
        "version": result.version,
        "setup": result.setup,
        "metric_split": metric_split,
        "model_name": result.model_name,
        "training_time_sec": result.training_time_sec,
        "accuracy": result.accuracy,
        "fraud_precision": result.fraud_precision,
        "fraud_recall": result.fraud_recall,
        "fraud_f1": result.fraud_f1,
        "macro_f1": result.macro_f1,
        "weighted_f1": result.weighted_f1,
        "roc_auc": result.roc_auc,
        "pr_auc": result.pr_auc,
        "best_params": json.dumps(result.best_params, default=str),
    }


def sorted_metrics_frame(results: List[EvalResult], metric_split: str) -> pd.DataFrame:
    rows = [result_to_row(result, metric_split) for result in results]
    return pd.DataFrame(rows).sort_values(
        by=["fraud_f1", "pr_auc", "fraud_recall", "fraud_precision", "roc_auc", "training_time_sec"],
        ascending=[False, False, False, False, False, True],
    )


def validate_overall_outputs(outdir: Path, versions: List[str]) -> None:
    overall_csv = outdir / "overall_best_models.csv"
    if not overall_csv.exists():
        raise FileNotFoundError(f"[VALIDATION ERROR] Missing overall summary: {overall_csv}")

    df = pd.read_csv(overall_csv)
    missing_versions = sorted(set(versions) - set(df["version"].astype(str)))
    if missing_versions:
        raise ValueError(f"[VALIDATION ERROR] Missing version(s) in {overall_csv}: {missing_versions}")

    print(f"[VALIDATION OK] {overall_csv} contains expected version rows.")


def evaluate_version(
    dataset_path: Path,
    version: str,
    use_smote: bool,
    out_root: Path,
    sample_size: int | None = None,
    quick: bool = False,
) -> Dict[str, object]:
    if not XGBOOST_AVAILABLE:
        raise RuntimeError("XGBoost is required for the full model list but is not available.")
    if not CATBOOST_AVAILABLE:
        raise RuntimeError("CatBoost is required for the full model list but is not available.")

    outdir = out_root / version
    ensure_dir(outdir)

    df = load_dataset(dataset_path)
    full_row_count = len(df)
    df = stratified_sample(df, sample_size)
    if "TransactionID" in df.columns:
        transaction_ids = df["TransactionID"].astype(str).reset_index(drop=True)
    else:
        transaction_ids = pd.Series(df.index.astype(str), name="TransactionID")

    X, y = prepare_features(df)

    X_train_val, X_test, y_train_val, y_test, ids_train_val, ids_test = train_test_split(
        X,
        y,
        transaction_ids,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    validation_fraction_of_train_val = VALIDATION_SIZE / (1.0 - TEST_SIZE)
    X_train, X_validation, y_train, y_validation, ids_train, ids_validation = train_test_split(
        X_train_val,
        y_train_val,
        ids_train_val,
        test_size=validation_fraction_of_train_val,
        random_state=RANDOM_STATE,
        stratify=y_train_val,
    )

    cv_splits = 2 if quick else 5
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    imbalance_ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    setup = "SMOTE" if use_smote else "No-SMOTE"
    validation_results: List[EvalResult] = []
    test_results: List[EvalResult] = []
    artifacts: Dict[str, object] = {}
    curve_data: Dict[str, Dict[str, object]] = {}

    specs = build_model_specs(
        use_smote=use_smote,
        imbalance_ratio=imbalance_ratio,
        quick=quick,
        feature_columns=X_train.columns.tolist(),
    )

    # sklearn-style models
    for spec in specs:
        print(f"[INFO] {version}: tuning {spec['name']} ...")
        best_estimator, best_params, elapsed = run_search_for_sklearn_model(
            spec=spec,
            X_train=X_train,
            y_train=y_train,
            cv=cv,
            scoring="average_precision",
        )

        validation_result, _, _ = evaluate_model_on_split(
            estimator=best_estimator,
            version=version,
            setup=setup,
            model_name=spec["name"],
            training_time_sec=elapsed,
            best_params=best_params,
            X_split=X_validation,
            y_split=y_validation,
            transaction_ids=ids_validation,
            split_name="validation",
            outdir=outdir,
        )
        test_result, y_pred, y_proba = evaluate_model_on_split(
            estimator=best_estimator,
            version=version,
            setup=setup,
            model_name=spec["name"],
            training_time_sec=elapsed,
            best_params=best_params,
            X_split=X_test,
            y_split=y_test,
            transaction_ids=ids_test,
            split_name="test",
            outdir=outdir,
        )

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)

        curve_data[spec["name"]] = {
            "roc_curve": (fpr, tpr),
            "pr_curve": (precision_curve, recall_curve),
            "roc_auc": test_result.roc_auc,
            "pr_auc": test_result.pr_auc,
        }

        plot_confusion_matrix(
            cm=np.array(test_result.confusion_matrix),
            model_name=spec["name"],
            version=version,
            save_path=outdir / f"confusion_matrix_{spec['name'].lower()}_{version}.png",
        )

        validation_results.append(validation_result)
        test_results.append(test_result)
        artifacts[spec["name"]] = best_estimator

    # CatBoost
    if CATBOOST_AVAILABLE:
        print(f"[INFO] {version}: tuning CatBoost ...")
        cb_model, cb_params, cb_elapsed = run_search_for_catboost(
            X_train=X_train,
            y_train=y_train,
            cv=cv,
            use_smote=use_smote,
            n_iter=2 if quick else 30,
            quick=quick,
        )

        validation_result, _, _ = evaluate_model_on_split(
            estimator=cb_model,
            version=version,
            setup=setup,
            model_name="CatBoost",
            training_time_sec=cb_elapsed,
            best_params=cb_params,
            X_split=X_validation,
            y_split=y_validation,
            transaction_ids=ids_validation,
            split_name="validation",
            outdir=outdir,
        )
        test_result, y_pred, y_proba = evaluate_model_on_split(
            estimator=cb_model,
            version=version,
            setup=setup,
            model_name="CatBoost",
            training_time_sec=cb_elapsed,
            best_params=cb_params,
            X_split=X_test,
            y_split=y_test,
            transaction_ids=ids_test,
            split_name="test",
            outdir=outdir,
        )

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)

        curve_data["CatBoost"] = {
            "roc_curve": (fpr, tpr),
            "pr_curve": (precision_curve, recall_curve),
            "roc_auc": test_result.roc_auc,
            "pr_auc": test_result.pr_auc,
        }

        plot_confusion_matrix(
            cm=np.array(test_result.confusion_matrix),
            model_name="CatBoost",
            version=version,
            save_path=outdir / f"confusion_matrix_catboost_{version}.png",
        )

        validation_results.append(validation_result)
        test_results.append(test_result)
        artifacts["CatBoost"] = cb_model

    completed_models = sorted(r.model_name for r in validation_results)
    missing_models = sorted(set(EXPECTED_MODEL_NAMES) - set(completed_models))
    if missing_models:
        raise RuntimeError(f"Missing completed model result(s) for {version}: {missing_models}")

    validation_metrics_df = sorted_metrics_frame(validation_results, "validation")
    test_metrics_df = sorted_metrics_frame(test_results, "test")
    validation_metrics_df.to_csv(outdir / "metrics_validation.csv", index=False)
    test_metrics_df.to_csv(outdir / "metrics_test.csv", index=False)
    test_metrics_df.to_csv(outdir / "metrics.csv", index=False)

    with open(outdir / "metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"====================== Model Evaluation ({version}) ======================\n\n")
        f.write(f"Dataset Path: {dataset_path}\n")
        f.write(f"Rows Used: {len(df)} of {full_row_count}\n")
        f.write(f"Setup: {setup}\n")
        f.write(f"Quick Mode: {quick}\n")
        f.write(f"Train Rows: {len(X_train)}\n")
        f.write(f"Validation Rows: {len(X_validation)}\n")
        f.write(f"Test Rows: {len(X_test)}\n")
        f.write("Label Convention: 0 = Legitimate, 1 = Fraud\n")
        f.write("Scoring Metric During Tuning: average_precision\n")
        f.write("Model Selection Split: validation\n")
        f.write("Final Reporting Split: test\n")
        f.write(f"Model Feature Columns: {', '.join(X_train.columns.tolist())}\n")
        dropped_columns = [c for c in EXCLUDED_MODEL_COLUMNS if c in df.columns]
        f.write(f"Excluded Columns: {', '.join(dropped_columns)}\n")
        f.write(f"Model List: {', '.join(EXPECTED_MODEL_NAMES)}\n")
        f.write("Model Selection Rule:\n")
        f.write(MODEL_SELECTION_RULE + "\n\n")
        for r in validation_results:
            test_match = next(t for t in test_results if t.model_name == r.model_name)
            f.write(f"=== {r.model_name} ===\n")
            f.write("[Validation metrics used for model selection]\n")
            f.write(r.report_text + "\n")
            f.write(f"Confusion Matrix:\n{np.array(r.confusion_matrix)}\n")
            f.write(f"Fraud Precision: {r.fraud_precision:.6f}\n")
            f.write(f"Fraud Recall: {r.fraud_recall:.6f}\n")
            f.write(f"Fraud F1: {r.fraud_f1:.6f}\n")
            f.write(f"ROC-AUC: {r.roc_auc:.6f}\n")
            f.write(f"PR-AUC: {r.pr_auc:.6f}\n")
            f.write(f"Training Time (sec): {r.training_time_sec:.2f}\n")
            f.write(f"Best Params: {json.dumps(r.best_params, default=str)}\n")
            f.write("\n[Final untouched test metrics]\n")
            f.write(test_match.report_text + "\n")
            f.write(f"Confusion Matrix:\n{np.array(test_match.confusion_matrix)}\n")
            f.write(f"Fraud Precision: {test_match.fraud_precision:.6f}\n")
            f.write(f"Fraud Recall: {test_match.fraud_recall:.6f}\n")
            f.write(f"Fraud F1: {test_match.fraud_f1:.6f}\n")
            f.write(f"ROC-AUC: {test_match.roc_auc:.6f}\n")
            f.write(f"PR-AUC: {test_match.pr_auc:.6f}\n")
            f.write("=" * 75 + "\n\n")

    plot_curves(curve_data, version, outdir)

    best_validation_result, selection_reason = select_best_model(validation_results)
    best_test_result = next(r for r in test_results if r.model_name == best_validation_result.model_name)
    save_best_model(
        best_result=best_validation_result,
        best_artifact=artifacts[best_validation_result.model_name],
        outdir=outdir,
        selection_reason=selection_reason,
        model_names=EXPECTED_MODEL_NAMES,
        dataset_path=dataset_path,
        sample_size=sample_size,
        quick=quick,
        final_test_result=best_test_result,
        train_rows=len(X_train),
        validation_rows=len(X_validation),
        test_rows=len(X_test),
        feature_columns=X_train.columns.tolist(),
        excluded_columns=[c for c in EXCLUDED_MODEL_COLUMNS if c in df.columns],
    )

    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "version": version,
                "dataset_path": str(dataset_path),
                "setup": "SMOTE" if use_smote else "No-SMOTE",
                "quick": quick,
                "sample_size": sample_size,
                "rows_used": len(df),
                "full_row_count": full_row_count,
                "train_rows": len(X_train),
                "validation_rows": len(X_validation),
                "test_rows": len(X_test),
                "cv_splits": cv_splits,
                "model_list": EXPECTED_MODEL_NAMES,
                "scoring_metric": "average_precision",
                "model_selection_split": "validation",
                "final_reporting_split": "test",
                "feature_columns": X_train.columns.tolist(),
                "excluded_columns": [c for c in EXCLUDED_MODEL_COLUMNS if c in df.columns],
                "selection_rule": MODEL_SELECTION_RULE,
                "selection_reason": selection_reason,
                "best_model": best_validation_result.model_name,
                "validation_metrics": [result_to_row(r, "validation") for r in validation_results],
                "test_metrics": [result_to_row(r, "test") for r in test_results],
            },
            f,
            indent=2,
        )

    validate_version_outputs(outdir, version, best_validation_result.model_name)

    print(
        f"[RESULT] {version} | best={best_validation_result.model_name} | "
        f"validation_f1={best_validation_result.fraud_f1:.4f} | "
        f"test_f1={best_test_result.fraud_f1:.4f} | "
        f"test_recall={best_test_result.fraud_recall:.4f} | "
        f"test_pr_auc={best_test_result.pr_auc:.4f}"
    )

    return {
        "version": version,
        "setup": setup,
        "model_name": best_validation_result.model_name,
        "selection_metric_source": "validation",
        "evaluation_metric_source": "test",
        "training_time_sec": best_test_result.training_time_sec,
        "accuracy": best_test_result.accuracy,
        "fraud_precision": best_test_result.fraud_precision,
        "fraud_recall": best_test_result.fraud_recall,
        "fraud_f1": best_test_result.fraud_f1,
        "macro_f1": best_test_result.macro_f1,
        "weighted_f1": best_test_result.weighted_f1,
        "roc_auc": best_test_result.roc_auc,
        "pr_auc": best_test_result.pr_auc,
        "validation_accuracy": best_validation_result.accuracy,
        "validation_fraud_precision": best_validation_result.fraud_precision,
        "validation_fraud_recall": best_validation_result.fraud_recall,
        "validation_fraud_f1": best_validation_result.fraud_f1,
        "validation_roc_auc": best_validation_result.roc_auc,
        "validation_pr_auc": best_validation_result.pr_auc,
        "selection_reason": selection_reason,
        "best_params": json.dumps(best_validation_result.best_params, default=str),
    }


def main():
    parser = argparse.ArgumentParser(description="Standard, cross-validated fraud-model experiments across v1 to v5.")
    parser.add_argument("--data_dir", default="data", help="Directory containing version CSV files")
    parser.add_argument("--outdir", default="results_standard", help="Directory to save outputs")
    parser.add_argument("--use_smote", action="store_true", help="Apply SMOTE within training folds")
    parser.add_argument(
        "--versions",
        nargs="+",
        help="Dataset versions to run, for example: --versions v1 or --versions v1 v2",
    )
    parser.add_argument("--sample_size", type=int, default=None, help="Optional stratified row sample per version")
    parser.add_argument("--quick", action="store_true", help="Use reduced CV/search budgets for smoke testing")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    requested_versions = parse_versions(args.versions)
    versions = find_version_files(data_dir, requested_versions)

    overall_rows = []
    for version, file_path in versions:
        print(f"\n[INFO] Running {version} using {file_path}")
        row = evaluate_version(
            dataset_path=file_path,
            version=version,
            use_smote=args.use_smote,
            out_root=outdir,
            sample_size=args.sample_size,
            quick=args.quick,
        )
        overall_rows.append(row)

    overall_df = pd.DataFrame(overall_rows)
    overall_df.to_csv(outdir / "overall_best_models.csv", index=False)
    validate_overall_outputs(outdir, [version for version, _ in versions])
    print(f"\n[OK] Overall summary saved to {outdir / 'overall_best_models.csv'}")


if __name__ == "__main__":
    main()
