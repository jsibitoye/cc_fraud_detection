from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

F1_TOLERANCE = 0.002
RECALL_TOLERANCE = 0.002
PRECISION_TOLERANCE = 0.002

REQUIRED_COLUMNS = {
    "version",
    "setup",
    "model_name",
    "training_time_sec",
    "fraud_precision",
    "fraud_recall",
    "fraud_f1",
    "roc_auc",
    "pr_auc",
}


def read_overall_csv(path: Path, expected_setup: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing comparison input CSV: {path}")

    df = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required column(s): {missing}")

    df = df.copy()
    df["version"] = df["version"].astype(str)
    if "setup" not in df.columns:
        df["setup"] = expected_setup
    return df


def describe_delta(delta: float, tolerance: float = F1_TOLERANCE) -> str:
    if abs(delta) <= tolerance:
        return "marginal/no practical change"
    if abs(delta) < 0.010:
        return "marginal improvement" if delta > 0 else "marginal decline"
    return "meaningful improvement" if delta > 0 else "meaningful decline"


def choose_version_winner(row: pd.Series) -> tuple[str, str, str]:
    f1_delta = row["fraud_f1_delta"]
    recall_delta = row["fraud_recall_delta"]
    precision_delta = row["fraud_precision_delta"]

    if f1_delta > F1_TOLERANCE:
        return "SMOTE", row["model_name_smote"], "SMOTE had higher Fraud F1."
    if f1_delta < -F1_TOLERANCE:
        return "No-SMOTE", row["model_name_no_smote"], "No-SMOTE had higher Fraud F1."
    if recall_delta > RECALL_TOLERANCE:
        return "SMOTE", row["model_name_smote"], "Fraud F1 was practically tied; SMOTE had higher fraud recall."
    if recall_delta < -RECALL_TOLERANCE:
        return "No-SMOTE", row["model_name_no_smote"], "Fraud F1 was practically tied; No-SMOTE had higher fraud recall."
    if precision_delta > PRECISION_TOLERANCE:
        return "SMOTE", row["model_name_smote"], "Fraud F1 and recall were practically tied; SMOTE had higher precision."
    if precision_delta < -PRECISION_TOLERANCE:
        return "No-SMOTE", row["model_name_no_smote"], "Fraud F1 and recall were practically tied; No-SMOTE had higher precision."

    no_smote_time = row["training_time_sec_no_smote"]
    smote_time = row["training_time_sec_smote"]
    if smote_time < no_smote_time:
        return "SMOTE", row["model_name_smote"], "Performance was practically tied; SMOTE trained faster."
    return "No-SMOTE", row["model_name_no_smote"], "Performance was practically tied; No-SMOTE trained faster or tied."


def plot_metric_comparison(df: pd.DataFrame, metric: str, ylabel: str, outfile: Path) -> None:
    versions = df["version"].tolist()
    x = np.arange(len(versions))
    width = 0.36

    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, df[f"{metric}_no_smote"], width, label="No-SMOTE")
    plt.bar(x + width / 2, df[f"{metric}_smote"], width, label="SMOTE")
    plt.xticks(x, versions)
    plt.ylabel(ylabel)
    plt.xlabel("Dataset version")
    plt.title(f"{ylabel}: SMOTE vs No-SMOTE")
    plt.ylim(bottom=0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()


def build_comparison(no_smote_df: pd.DataFrame, smote_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = no_smote_df.merge(
        smote_df,
        on="version",
        suffixes=("_no_smote", "_smote"),
        how="outer",
        indicator=True,
    )

    unmatched = merged[merged["_merge"] != "both"]["version"].tolist()
    if unmatched:
        raise ValueError(f"SMOTE and No-SMOTE summaries do not cover the same version(s): {unmatched}")

    merged = merged.drop(columns=["_merge"]).sort_values("version").reset_index(drop=True)

    for metric in ["fraud_f1", "fraud_recall", "fraud_precision", "roc_auc", "pr_auc"]:
        merged[f"{metric}_delta"] = merged[f"{metric}_smote"] - merged[f"{metric}_no_smote"]

    merged["smote_improved_fraud_f1"] = merged["fraud_f1_delta"] > F1_TOLERANCE
    merged["smote_improved_fraud_recall"] = merged["fraud_recall_delta"] > RECALL_TOLERANCE
    merged["smote_hurt_precision"] = merged["fraud_precision_delta"] < -PRECISION_TOLERANCE
    merged["f1_effect_size"] = merged["fraud_f1_delta"].map(describe_delta)

    winner_rows = []
    for _, row in merged.iterrows():
        setup, model, rationale = choose_version_winner(row)
        winner_rows.append({
            "version": row["version"],
            "no_smote_best_model": row["model_name_no_smote"],
            "smote_best_model": row["model_name_smote"],
            "winning_setup": setup,
            "winning_model": model,
            "fraud_f1_delta_smote_minus_no_smote": row["fraud_f1_delta"],
            "fraud_recall_delta_smote_minus_no_smote": row["fraud_recall_delta"],
            "fraud_precision_delta_smote_minus_no_smote": row["fraud_precision_delta"],
            "smote_improved_fraud_f1": row["smote_improved_fraud_f1"],
            "smote_improved_fraud_recall": row["smote_improved_fraud_recall"],
            "smote_hurt_precision": row["smote_hurt_precision"],
            "effect_size": row["f1_effect_size"],
            "rationale": rationale,
        })

    return merged, pd.DataFrame(winner_rows)


def final_recommendation(no_smote_df: pd.DataFrame, smote_df: pd.DataFrame) -> pd.Series:
    combined = pd.concat([no_smote_df, smote_df], ignore_index=True)
    return combined.sort_values(
        by=["fraud_f1", "pr_auc", "fraud_recall", "fraud_precision", "roc_auc", "training_time_sec"],
        ascending=[False, False, False, False, False, True],
    ).iloc[0]


def write_summary(
    comparison_df: pd.DataFrame,
    winner_df: pd.DataFrame,
    recommendation: pd.Series,
    outfile: Path,
) -> None:
    with open(outfile, "w", encoding="utf-8") as f:
        f.write("SMOTE vs No-SMOTE Comparison Summary\n")
        f.write("===================================\n\n")
        f.write("Selection basis: each training run selects its model family on the validation split. ")
        f.write("This comparison uses the selected models' untouched test metrics, ranked by Fraud F1 first, ")
        f.write("then fraud recall, precision, PR-AUC, ROC-AUC, and training time for practical ties. ")
        f.write("Accuracy is not used for setup recommendation.\n\n")

        for _, row in comparison_df.iterrows():
            winner = winner_df[winner_df["version"] == row["version"]].iloc[0]
            f.write(f"Version {row['version']}:\n")
            f.write(f"- No-SMOTE best model: {row['model_name_no_smote']} ")
            f.write(f"(Fraud F1={row['fraud_f1_no_smote']:.6f}, Recall={row['fraud_recall_no_smote']:.6f}, ")
            f.write(f"Precision={row['fraud_precision_no_smote']:.6f})\n")
            f.write(f"- SMOTE best model: {row['model_name_smote']} ")
            f.write(f"(Fraud F1={row['fraud_f1_smote']:.6f}, Recall={row['fraud_recall_smote']:.6f}, ")
            f.write(f"Precision={row['fraud_precision_smote']:.6f})\n")
            f.write(f"- SMOTE Fraud F1 delta: {row['fraud_f1_delta']:.6f} ({row['f1_effect_size']})\n")
            f.write(f"- SMOTE recall delta: {row['fraud_recall_delta']:.6f}; ")
            f.write(f"precision delta: {row['fraud_precision_delta']:.6f}\n")
            f.write(f"- Winner: {winner['winning_setup']} / {winner['winning_model']} - {winner['rationale']}\n\n")

        f.write("Final Recommended Model/Setup\n")
        f.write("-----------------------------\n")
        f.write(f"Version: {recommendation['version']}\n")
        f.write(f"Setup: {recommendation['setup']}\n")
        f.write(f"Model: {recommendation['model_name']}\n")
        f.write(f"Fraud F1: {recommendation['fraud_f1']:.6f}\n")
        f.write(f"Fraud Recall: {recommendation['fraud_recall']:.6f}\n")
        f.write(f"Fraud Precision: {recommendation['fraud_precision']:.6f}\n")
        f.write(f"PR-AUC: {recommendation['pr_auc']:.6f}\n")
        f.write(f"ROC-AUC: {recommendation['roc_auc']:.6f}\n\n")
        f.write("Threshold tuning uses validation predictions to choose an operating point and reports that threshold on test predictions.\n")


def validate_outputs(outdir: Path) -> None:
    required = [
        outdir / "smote_vs_no_smote_comparison.csv",
        outdir / "version_winner_table.csv",
        outdir / "comparison_summary.txt",
        outdir / "fraud_f1_comparison.png",
        outdir / "fraud_recall_comparison.png",
        outdir / "fraud_precision_comparison.png",
        outdir / "pr_auc_comparison.png",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("[VALIDATION ERROR] Missing comparison output(s):\n" + "\n".join(missing))
    print(f"[VALIDATION OK] {outdir} contains all required comparison outputs.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare No-SMOTE and SMOTE experiment summaries.")
    parser.add_argument("--no_smote", required=True, help="Path to No-SMOTE overall_best_models.csv")
    parser.add_argument("--smote", required=True, help="Path to SMOTE overall_best_models.csv")
    parser.add_argument("--outdir", default="results_comparison", help="Directory for comparison outputs")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    no_smote_df = read_overall_csv(Path(args.no_smote), "No-SMOTE")
    smote_df = read_overall_csv(Path(args.smote), "SMOTE")
    comparison_df, winner_df = build_comparison(no_smote_df, smote_df)
    recommendation = final_recommendation(no_smote_df, smote_df)

    comparison_df.to_csv(outdir / "smote_vs_no_smote_comparison.csv", index=False)
    winner_df.to_csv(outdir / "version_winner_table.csv", index=False)

    plot_metric_comparison(comparison_df, "fraud_f1", "Fraud F1", outdir / "fraud_f1_comparison.png")
    plot_metric_comparison(comparison_df, "fraud_recall", "Fraud Recall", outdir / "fraud_recall_comparison.png")
    plot_metric_comparison(comparison_df, "fraud_precision", "Fraud Precision", outdir / "fraud_precision_comparison.png")
    plot_metric_comparison(comparison_df, "pr_auc", "PR-AUC", outdir / "pr_auc_comparison.png")

    write_summary(comparison_df, winner_df, recommendation, outdir / "comparison_summary.txt")
    validate_outputs(outdir)
    print(f"[OK] Comparison outputs saved to {outdir}")


if __name__ == "__main__":
    main()
