from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from compare_experiment_results import validate_outputs as validate_comparison_outputs
from run_experiments_standard import validate_version_outputs
from tune_thresholds import validate_threshold_outputs


def run_cmd(cmd: list[str]) -> None:
    print("\n[RUN]", " ".join(cmd), flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(f"[ERROR] Command failed with exit code {result.returncode}")


def validate_training_root(root: Path, label: str) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"{label} results root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"{label} results root is not a directory: {root}")

    overall_csv = root / "overall_best_models.csv"
    if not overall_csv.exists():
        raise FileNotFoundError(f"{label} missing overall_best_models.csv: {overall_csv}")

    overall_df = pd.read_csv(overall_csv)
    if "version" not in overall_df.columns or "model_name" not in overall_df.columns:
        raise ValueError(f"{overall_csv} must contain version and model_name columns.")

    version_dirs: list[Path] = []
    for _, row in overall_df.iterrows():
        version = str(row["version"])
        best_model_name = str(row["model_name"])
        version_dir = root / version
        validate_version_outputs(version_dir, version, best_model_name)
        version_dirs.append(version_dir)

    print(f"[VALIDATION OK] {label} root is ready for post-training analysis: {root}", flush=True)
    return version_dirs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run threshold tuning and comparison in one pass.")
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument("--no_smote_root", default="results_standard")
    parser.add_argument("--smote_root", default="results_standard_smote")
    parser.add_argument("--threshold_mode", choices=["max_f1", "recall_constrained"], default="max_f1")
    parser.add_argument("--min_recall", type=float, default=0.85)
    parser.add_argument("--no_smote_threshold_outdir", default="results_threshold_tuning_no_smote")
    parser.add_argument("--smote_threshold_outdir", default="results_threshold_tuning_smote")
    parser.add_argument("--compare_no_smote_csv", default=None)
    parser.add_argument("--compare_smote_csv", default=None)
    parser.add_argument("--comparison_outdir", default="results_comparison")
    args = parser.parse_args()

    py = args.python_bin
    no_smote_root = Path(args.no_smote_root)
    smote_root = Path(args.smote_root)
    no_smote_threshold_outdir = Path(args.no_smote_threshold_outdir)
    smote_threshold_outdir = Path(args.smote_threshold_outdir)
    comparison_outdir = Path(args.comparison_outdir)
    compare_no_smote_csv = Path(args.compare_no_smote_csv) if args.compare_no_smote_csv else no_smote_root / "overall_best_models.csv"
    compare_smote_csv = Path(args.compare_smote_csv) if args.compare_smote_csv else smote_root / "overall_best_models.csv"

    no_smote_version_dirs = validate_training_root(no_smote_root, "No-SMOTE")
    smote_version_dirs = validate_training_root(smote_root, "SMOTE")

    # Threshold tuning: no-SMOTE
    run_cmd([
        py,
        "scripts/tune_thresholds.py",
        "--predictions_root", str(no_smote_root),
        "--outdir", str(no_smote_threshold_outdir),
        "--mode", args.threshold_mode,
        "--min_recall", str(args.min_recall),
    ])
    validate_threshold_outputs(no_smote_threshold_outdir, no_smote_version_dirs)

    # Threshold tuning: SMOTE
    run_cmd([
        py,
        "scripts/tune_thresholds.py",
        "--predictions_root", str(smote_root),
        "--outdir", str(smote_threshold_outdir),
        "--mode", args.threshold_mode,
        "--min_recall", str(args.min_recall),
    ])
    validate_threshold_outputs(smote_threshold_outdir, smote_version_dirs)

    # Compare overall best model summaries
    run_cmd([
        py,
        "scripts/compare_experiment_results.py",
        "--no_smote", str(compare_no_smote_csv),
        "--smote", str(compare_smote_csv),
        "--outdir", str(comparison_outdir),
    ])
    validate_comparison_outputs(comparison_outdir)

    print("\n[OK] Full post-training pipeline completed successfully.")


if __name__ == "__main__":
    main()
