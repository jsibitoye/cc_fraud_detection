# scripts/tune_thresholds.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

EXPECTED_MODEL_NAMES = ["LogisticRegression", "DecisionTree", "RandomForest", "XGBoost", "CatBoost"]
REQUIRED_PREDICTION_COLUMNS = {"y_true", "y_pred", "y_proba"}


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> dict:
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def choose_best_threshold(
    metrics_df: pd.DataFrame,
    mode: str,
    min_recall: float,
) -> tuple[pd.Series, str]:
    """
    Returns:
        best_row, selection_note
    """

    best_f1_row = metrics_df.sort_values(
        by=["f1", "recall", "precision", "accuracy"],
        ascending=[False, False, False, False],
    ).iloc[0]

    if mode == "max_f1":
        return best_f1_row, "Selected threshold with maximum fraud F1 over the threshold grid."

    constrained = metrics_df[metrics_df["recall"] >= min_recall].copy()

    if mode == "recall_constrained":
        if not constrained.empty:
            best_row = constrained.sort_values(
                by=["f1", "precision", "accuracy"],
                ascending=[False, False, False],
            ).iloc[0]
            note = f"Selected best F1 subject to recall >= {min_recall:.3f}"
        else:
            best_row = metrics_df.sort_values(
                by=["recall", "f1", "precision", "accuracy"],
                ascending=[False, False, False, False],
            ).iloc[0]
            note = (
                f"No threshold achieved recall >= {min_recall:.3f}. "
                f"Fell back to maximum-recall operating point."
            )
        return best_row, note

    raise ValueError(f"Unsupported mode: {mode}")


def plot_threshold_curves(metrics_df: pd.DataFrame, title: str, outfile: Path) -> None:
    plt.figure(figsize=(11, 7))
    plt.plot(metrics_df["threshold"], metrics_df["precision"], label="Precision", linewidth=2)
    plt.plot(metrics_df["threshold"], metrics_df["recall"], label="Recall", linewidth=2)
    plt.plot(metrics_df["threshold"], metrics_df["f1"], label="F1", linewidth=2)
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()


def plot_confusion_matrix(best_row: pd.Series, title: str, outfile: Path) -> None:
    cm = np.array([
        [best_row["tn"], best_row["fp"]],
        [best_row["fn"], best_row["tp"]],
    ])

    plt.figure(figsize=(8, 7))
    plt.imshow(cm, cmap="viridis")
    plt.colorbar()

    labels = ["Legitimate", "Fraud"]
    plt.xticks([0, 1], labels)
    plt.yticks([0, 1], labels)
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.title(title)

    max_val = cm.max() if cm.size else 1
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > (max_val / 2) else "black"
            plt.text(j, i, f"{cm[i, j]}", ha="center", va="center", color=color, fontsize=22, fontweight="bold")

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()


def process_prediction_pair(
    selection_prediction_file: Path,
    evaluation_prediction_file: Path,
    version: str,
    model_name: str,
    outdir: Path,
    mode: str,
    min_recall: float,
    thresholds: np.ndarray,
    selection_split: str,
    evaluation_split: str,
) -> dict:
    selection_df = pd.read_csv(selection_prediction_file)

    missing_cols = sorted(REQUIRED_PREDICTION_COLUMNS - set(selection_df.columns))
    if missing_cols:
        raise ValueError(f"{selection_prediction_file} is missing required column(s): {missing_cols}")

    evaluation_df = pd.read_csv(evaluation_prediction_file)
    missing_cols = sorted(REQUIRED_PREDICTION_COLUMNS - set(evaluation_df.columns))
    if missing_cols:
        raise ValueError(f"{evaluation_prediction_file} is missing required column(s): {missing_cols}")

    y_selection_true = selection_df["y_true"].to_numpy().astype(int)
    y_selection_proba = selection_df["y_proba"].to_numpy().astype(float)
    y_evaluation_true = evaluation_df["y_true"].to_numpy().astype(int)
    y_evaluation_proba = evaluation_df["y_proba"].to_numpy().astype(float)

    if len(y_selection_true) == 0:
        raise ValueError(f"Empty selection prediction file: {selection_prediction_file}")
    if len(y_evaluation_true) == 0:
        raise ValueError(f"Empty evaluation prediction file: {evaluation_prediction_file}")

    metrics = [compute_metrics(y_selection_true, y_selection_proba, t) for t in thresholds]
    metrics_df = pd.DataFrame(metrics)

    best_row, selection_note = choose_best_threshold(metrics_df, mode=mode, min_recall=min_recall)
    evaluation_row = compute_metrics(
        y_evaluation_true,
        y_evaluation_proba,
        float(best_row["threshold"]),
    )

    model_outdir = outdir / version / model_name
    model_outdir.mkdir(parents=True, exist_ok=True)

    metrics_csv = model_outdir / "threshold_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    pd.DataFrame([evaluation_row]).to_csv(model_outdir / "test_metrics_at_selected_threshold.csv", index=False)

    best_payload = {
        "version": version,
        "model_name": model_name,
        "selection_source_file": str(selection_prediction_file),
        "evaluation_source_file": str(evaluation_prediction_file),
        "selection_split": selection_split,
        "evaluation_split": evaluation_split,
        "selection_mode": mode,
        "min_recall": float(min_recall),
        "selection_note": selection_note,
        "best_threshold": float(best_row["threshold"]),
        "selection_accuracy": float(best_row["accuracy"]),
        "selection_precision": float(best_row["precision"]),
        "selection_recall": float(best_row["recall"]),
        "selection_f1": float(best_row["f1"]),
        "selection_tn": int(best_row["tn"]),
        "selection_fp": int(best_row["fp"]),
        "selection_fn": int(best_row["fn"]),
        "selection_tp": int(best_row["tp"]),
        "accuracy": float(evaluation_row["accuracy"]),
        "precision": float(evaluation_row["precision"]),
        "recall": float(evaluation_row["recall"]),
        "f1": float(evaluation_row["f1"]),
        "tn": int(evaluation_row["tn"]),
        "fp": int(evaluation_row["fp"]),
        "fn": int(evaluation_row["fn"]),
        "tp": int(evaluation_row["tp"]),
    }

    with open(model_outdir / "best_threshold.json", "w", encoding="utf-8") as f:
        json.dump(best_payload, f, indent=2)

    plot_threshold_curves(
        metrics_df,
        title=f"{model_name} Threshold Optimization ({version}, validation)",
        outfile=model_outdir / "threshold_curve.png",
    )

    plot_confusion_matrix(
        pd.Series(evaluation_row),
        title=f"{model_name} Tuned Test Confusion Matrix ({version}, t={best_row['threshold']:.3f})",
        outfile=model_outdir / "confusion_matrix_tuned.png",
    )

    print(
        f"[OK] {version}/{model_name} | "
        f"threshold={best_row['threshold']:.3f} | "
        f"validation_f1={best_row['f1']:.4f} | "
        f"test_precision={evaluation_row['precision']:.4f} | "
        f"test_recall={evaluation_row['recall']:.4f} | "
        f"test_f1={evaluation_row['f1']:.4f}"
    )
    if "fell back" in selection_note.lower():
        print(f"[WARN] {version}/{model_name}: {selection_note}")
    else:
        print(f"[INFO] {version}/{model_name}: {selection_note}")

    return best_payload


def normalize_mode(mode: str) -> str:
    if mode == "target_recall":
        return "recall_constrained"
    return mode


def validate_prediction_inputs(
    predictions_root: Path,
    selection_split: str = "validation",
    evaluation_split: str = "test",
) -> list[Path]:
    if not predictions_root.exists():
        raise FileNotFoundError(f"Predictions root does not exist: {predictions_root}")
    if not predictions_root.is_dir():
        raise NotADirectoryError(f"Predictions root is not a directory: {predictions_root}")

    version_dirs = sorted([p for p in predictions_root.iterdir() if p.is_dir() and p.name.startswith("v")])
    if not version_dirs:
        raise FileNotFoundError(f"No version folders found in predictions root: {predictions_root}")

    missing_files: list[str] = []
    missing_columns: list[str] = []

    for version_dir in version_dirs:
        for model_name in EXPECTED_MODEL_NAMES:
            for split in [selection_split, evaluation_split]:
                prediction_file = version_dir / f"{model_name}_{split}_predictions.csv"
                if not prediction_file.exists():
                    missing_files.append(str(prediction_file))
                    continue

                cols = set(pd.read_csv(prediction_file, nrows=0).columns)
                missing = sorted(REQUIRED_PREDICTION_COLUMNS - cols)
                if missing:
                    missing_columns.append(f"{prediction_file}: missing {missing}")

    if missing_files:
        raise FileNotFoundError(
            "[VALIDATION ERROR] Missing prediction CSV file(s):\n" + "\n".join(missing_files)
        )
    if missing_columns:
        raise ValueError(
            "[VALIDATION ERROR] Prediction CSV schema problem(s):\n" + "\n".join(missing_columns)
        )

    print(
        f"[VALIDATION OK] Found required {selection_split} and {evaluation_split} "
        f"prediction CSVs in {predictions_root}"
    )
    return version_dirs


def validate_threshold_outputs(outdir: Path, version_dirs: list[Path]) -> None:
    required_top = [outdir / "all_best_thresholds.csv", outdir / "best_thresholds_by_version.csv"]
    missing = [str(path) for path in required_top if not path.exists()]

    for version_dir in version_dirs:
        version = version_dir.name
        for model_name in EXPECTED_MODEL_NAMES:
            model_outdir = outdir / version / model_name
            for filename in [
                "threshold_metrics.csv",
                "test_metrics_at_selected_threshold.csv",
                "threshold_curve.png",
                "confusion_matrix_tuned.png",
                "best_threshold.json",
            ]:
                path = model_outdir / filename
                if not path.exists():
                    missing.append(str(path))

    if missing:
        raise FileNotFoundError(
            "[VALIDATION ERROR] Missing threshold output file(s):\n" + "\n".join(missing)
        )

    print(f"[VALIDATION OK] {outdir} contains all required threshold tuning outputs.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_root", required=True, help="Root folder containing version folders")
    parser.add_argument("--outdir", required=True, help="Output folder for threshold tuning results")
    parser.add_argument(
        "--selection_split",
        default="validation",
        help="Prediction split used to choose thresholds",
    )
    parser.add_argument(
        "--evaluation_split",
        default="test",
        help="Prediction split used only to evaluate chosen thresholds",
    )
    parser.add_argument(
        "--mode",
        choices=["max_f1", "recall_constrained", "target_recall"],
        default="max_f1",
        help="Selection strategy",
    )
    parser.add_argument(
        "--min_recall",
        type=float,
        default=0.85,
        help="Recall target used by selection logic",
    )
    parser.add_argument(
        "--threshold_start",
        type=float,
        default=0.10,
        help="Start of threshold grid",
    )
    parser.add_argument(
        "--threshold_end",
        type=float,
        default=0.90,
        help="End of threshold grid",
    )
    parser.add_argument(
        "--threshold_step",
        type=float,
        default=0.01,
        help="Step size for threshold grid",
    )
    args = parser.parse_args()

    predictions_root = Path(args.predictions_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    mode = normalize_mode(args.mode)

    if args.threshold_step <= 0:
        raise ValueError("--threshold_step must be positive.")
    if args.threshold_start >= args.threshold_end:
        raise ValueError("--threshold_start must be less than --threshold_end.")

    thresholds = np.round(
        np.arange(args.threshold_start, args.threshold_end + args.threshold_step, args.threshold_step),
        3,
    )

    version_dirs = validate_prediction_inputs(
        predictions_root,
        selection_split=args.selection_split,
        evaluation_split=args.evaluation_split,
    )

    all_results: list[dict] = []

    for version_dir in version_dirs:
        version = version_dir.name

        for model_name in EXPECTED_MODEL_NAMES:
            selection_prediction_file = version_dir / f"{model_name}_{args.selection_split}_predictions.csv"
            evaluation_prediction_file = version_dir / f"{model_name}_{args.evaluation_split}_predictions.csv"
            result = process_prediction_pair(
                selection_prediction_file=selection_prediction_file,
                evaluation_prediction_file=evaluation_prediction_file,
                version=version,
                model_name=model_name,
                outdir=outdir,
                mode=mode,
                min_recall=args.min_recall,
                thresholds=thresholds,
                selection_split=args.selection_split,
                evaluation_split=args.evaluation_split,
            )
            all_results.append(result)

    if not all_results:
        raise RuntimeError("No threshold tuning results were generated.")

    all_results_df = pd.DataFrame(all_results)
    all_results_df.to_csv(outdir / "all_best_thresholds.csv", index=False)

    best_by_version = (
        all_results_df.sort_values(
            by=["version", "selection_f1", "selection_precision", "selection_recall", "selection_accuracy"],
            ascending=[True, False, False, False, False],
        )
        .groupby("version", as_index=False)
        .first()
    )
    best_by_version.to_csv(outdir / "best_thresholds_by_version.csv", index=False)

    validate_threshold_outputs(outdir, version_dirs)
    print(f"[OK] Saved all threshold results to: {outdir / 'all_best_thresholds.csv'}")
    print(f"[OK] Saved best-per-version summary to: {outdir / 'best_thresholds_by_version.csv'}")


if __name__ == "__main__":
    main()
