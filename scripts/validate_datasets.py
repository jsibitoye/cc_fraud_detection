from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

BASE_COLUMNS = ["TransactionID", "Amount", "Merchant", "Category", "CardType", "Location", "Time", "FraudFlag"]
REQUIRED_COLUMNS_BY_VERSION = {
    "v1": BASE_COLUMNS,
    "v2": BASE_COLUMNS + ["HighAmountFlag", "IsNightTransaction", "SubCategory"],
    "v3": BASE_COLUMNS + ["HighAmountFlag", "IsNightTransaction", "SubCategory", "MerchantRisk", "CardRisk"],
    "v4": BASE_COLUMNS
    + [
        "HighAmountFlag",
        "IsNightTransaction",
        "SubCategory",
        "MerchantRisk",
        "CardRisk",
        "Hour",
        "DayOfWeek",
        "Month",
        "IsWeekend",
        "AmountLog",
        "AmountToCategoryMedian",
        "AmountZScoreByCategory",
        "AmountPercentile",
    ],
    "v5": BASE_COLUMNS
    + [
        "HighAmountFlag",
        "IsNightTransaction",
        "SubCategory",
        "MerchantRisk",
        "CardRisk",
        "Hour",
        "DayOfWeek",
        "Month",
        "IsWeekend",
        "AmountLog",
        "AmountToCategoryMedian",
        "AmountZScoreByCategory",
        "AmountPercentile",
        "NightHighAmountFlag",
        "RiskScoreComposite",
    ],
}


def validate_dataset(path: Path, version: str, min_fraud_ratio: float, max_fraud_ratio: float) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required dataset file: {path}")

    df = pd.read_csv(path)
    required = REQUIRED_COLUMNS_BY_VERSION[version]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required column(s): {missing}")

    extra_missing = [col for col in df.columns if df[col].isna().any()]
    if extra_missing:
        raise ValueError(f"{path} contains missing values in column(s): {extra_missing}")

    fraud_values = set(pd.to_numeric(df["FraudFlag"], errors="coerce").dropna().unique().tolist())
    if not fraud_values.issubset({0, 1}):
        raise ValueError(f"{path} has non-binary FraudFlag values: {sorted(fraud_values)}")

    if df["FraudFlag"].isna().any():
        raise ValueError(f"{path} contains missing FraudFlag values")

    fraud_ratio = float(df["FraudFlag"].mean())
    if fraud_ratio < min_fraud_ratio or fraud_ratio > max_fraud_ratio:
        raise ValueError(
            f"{path} has unreasonable fraud ratio {fraud_ratio:.6f}; "
            f"expected between {min_fraud_ratio:.3f} and {max_fraud_ratio:.3f}"
        )

    exact_duplicates = int(df.duplicated().sum())
    if exact_duplicates:
        raise ValueError(f"{path} contains {exact_duplicates} exact duplicate row(s)")

    duplicate_ids = int(df["TransactionID"].duplicated().sum())
    if duplicate_ids:
        raise ValueError(f"{path} contains {duplicate_ids} duplicate TransactionID value(s)")

    amount = pd.to_numeric(df["Amount"], errors="coerce")
    if amount.isna().any():
        raise ValueError(f"{path} contains non-numeric Amount values")
    if (amount <= 0).any():
        raise ValueError(f"{path} contains non-positive Amount values")
    if (amount > 10_000_000).any():
        raise ValueError(f"{path} contains unrealistic Amount values above 10,000,000")

    parsed_time = pd.to_datetime(df["Time"], errors="coerce", format="mixed")
    if parsed_time.isna().any():
        raise ValueError(f"{path} contains invalid Time values")

    train_ids, test_ids = train_test_split(
        df["TransactionID"],
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=df["FraudFlag"],
    )
    overlap = set(train_ids).intersection(set(test_ids))
    if overlap:
        raise ValueError(f"{path} has train/test TransactionID overlap under validation split: {len(overlap)}")

    return {
        "version": version,
        "path": str(path),
        "rows": len(df),
        "columns": len(df.columns),
        "fraud_count": int(df["FraudFlag"].sum()),
        "legitimate_count": int((df["FraudFlag"] == 0).sum()),
        "fraud_ratio": round(fraud_ratio, 6),
        "unique_transaction_ids": int(df["TransactionID"].nunique()),
        "status": "PASS",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate final research fraud datasets before training.")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--min_fraud_ratio", type=float, default=0.01)
    parser.add_argument("--max_fraud_ratio", type=float, default=0.40)
    parser.add_argument("--summary_out", default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    results = []
    for version in ["v1", "v2", "v3", "v4", "v5"]:
        result = validate_dataset(
            data_dir / f"dataset_{version}.csv",
            version,
            min_fraud_ratio=args.min_fraud_ratio,
            max_fraud_ratio=args.max_fraud_ratio,
        )
        results.append(result)
        print(
            f"[PASS] {version}: rows={result['rows']} columns={result['columns']} "
            f"fraud_ratio={result['fraud_ratio']:.6f}"
        )

    summary = pd.DataFrame(results)
    if args.summary_out:
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_path, index=False)
        print(f"[OK] Validation summary saved to {summary_path}")

    print("[OK] All five final datasets passed validation.")


if __name__ == "__main__":
    main()
