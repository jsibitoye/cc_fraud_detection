import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


FINAL_COLUMNS = [
    "Amount",
    "Merchant",
    "Category",
    "CardType",
    "Location",
    "Time",
    "FraudFlag",
    "HighAmountFlag",
    "IsNightTransaction",
    "MerchantRisk",
    "CardRisk",
]

BINARY_COLUMNS = ["FraudFlag", "HighAmountFlag", "IsNightTransaction"]
NUMERIC_COLUMNS = ["Amount", "MerchantRisk", "CardRisk"]
CATEGORICAL_COLUMNS = ["Merchant", "Category", "CardType", "Location"]


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        clean = col.strip()

        if clean in ["Amount_₦", "Amount(NGN)", "TransactionAmount", "amount"]:
            rename_map[col] = "Amount"
        elif clean.lower() == "merchant":
            rename_map[col] = "Merchant"
        elif clean.lower() == "category":
            rename_map[col] = "Category"
        elif clean.lower() == "cardtype":
            rename_map[col] = "CardType"
        elif clean.lower() == "location":
            rename_map[col] = "Location"
        elif clean.lower() == "time":
            rename_map[col] = "Time"
        elif clean.lower() in ["fraudflag", "isfraud", "fraud"]:
            rename_map[col] = "FraudFlag"
        elif clean.lower() == "highamountflag":
            rename_map[col] = "HighAmountFlag"
        elif clean.lower() == "isnighttransaction":
            rename_map[col] = "IsNightTransaction"
        elif clean.lower() == "merchantrisk":
            rename_map[col] = "MerchantRisk"
        elif clean.lower() == "cardrisk":
            rename_map[col] = "CardRisk"

    df = df.rename(columns=rename_map)
    df.columns = [c.strip() for c in df.columns]
    return df


def ensure_final_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in FINAL_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[FINAL_COLUMNS]


def parse_time_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    parsed = pd.to_datetime(df["Time"], errors="coerce")
    df["Time"] = parsed
    return df


def fill_amount(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")

    # Fill by category median first, then global median
    global_median = df["Amount"].median()
    category_medians = df.groupby("Category")["Amount"].median()

    missing_mask = df["Amount"].isna()
    if missing_mask.any():
        df.loc[missing_mask, "Amount"] = df.loc[missing_mask, "Category"].map(category_medians)

    df["Amount"] = df["Amount"].fillna(global_median)

    # Guard against invalid values
    df["Amount"] = df["Amount"].clip(lower=0.01)
    return df


def fill_categorical(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].astype("string")
        df[col] = df[col].replace(
            to_replace=["", "nan", "None", "<NA>"],
            value=pd.NA
        )

    # Use mode where possible, otherwise "Unknown"
    for col in CATEGORICAL_COLUMNS:
        mode_val = df[col].mode(dropna=True)
        fill_val = mode_val.iloc[0] if not mode_val.empty else "Unknown"
        df[col] = df[col].fillna(fill_val)

    return df


def derive_high_amount_flag(df: pd.DataFrame) -> pd.Series:
    threshold = df["Amount"].quantile(0.90)
    return (df["Amount"] >= threshold).astype(int)


def derive_is_night_transaction(df: pd.DataFrame) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(df["Time"]):
        hours = df["Time"].dt.hour.fillna(12).astype(int)
        return hours.isin([0, 1, 2, 3, 4, 5, 23]).astype(int)
    return pd.Series(np.zeros(len(df), dtype=int), index=df.index)


def fill_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # FraudFlag
    df["FraudFlag"] = pd.to_numeric(df["FraudFlag"], errors="coerce")
    df["FraudFlag"] = df["FraudFlag"].fillna(0)
    df["FraudFlag"] = df["FraudFlag"].apply(lambda x: 1 if x >= 0.5 else 0).astype(int)

    # HighAmountFlag
    current = pd.to_numeric(df["HighAmountFlag"], errors="coerce")
    derived = derive_high_amount_flag(df)
    df["HighAmountFlag"] = current.where(~current.isna(), derived)
    df["HighAmountFlag"] = df["HighAmountFlag"].apply(lambda x: 1 if float(x) >= 0.5 else 0).astype(int)

    # IsNightTransaction
    current = pd.to_numeric(df["IsNightTransaction"], errors="coerce")
    derived = derive_is_night_transaction(df)
    df["IsNightTransaction"] = current.where(~current.isna(), derived)
    df["IsNightTransaction"] = df["IsNightTransaction"].apply(lambda x: 1 if float(x) >= 0.5 else 0).astype(int)

    return df


def fill_risk_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # MerchantRisk: fraud rate by merchant
    merchant_risk_existing = pd.to_numeric(df["MerchantRisk"], errors="coerce")
    merchant_rates = df.groupby("Merchant")["FraudFlag"].mean()
    merchant_fill = df["Merchant"].map(merchant_rates)

    # CardRisk: fraud rate by card type
    card_risk_existing = pd.to_numeric(df["CardRisk"], errors="coerce")
    card_rates = df.groupby("CardType")["FraudFlag"].mean()
    card_fill = df["CardType"].map(card_rates)

    global_fraud_rate = df["FraudFlag"].mean()

    df["MerchantRisk"] = merchant_risk_existing.where(~merchant_risk_existing.isna(), merchant_fill)
    df["CardRisk"] = card_risk_existing.where(~card_risk_existing.isna(), card_fill)

    df["MerchantRisk"] = df["MerchantRisk"].fillna(global_fraud_rate).clip(0, 1)
    df["CardRisk"] = df["CardRisk"].fillna(global_fraud_rate).clip(0, 1)

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    before = len(df)
    df = df.drop_duplicates()

    # Also drop duplicates on key transaction fields if exact row duplicates are not enough
    subset_cols = ["Amount", "Merchant", "Category", "CardType", "Location", "Time", "FraudFlag"]
    subset_cols = [c for c in subset_cols if c in df.columns]
    df = df.drop_duplicates(subset=subset_cols)

    after = len(df)
    print(f"[INFO] Removed {before - after} duplicate rows")
    return df


def create_missing_value_report(before_missing: pd.Series, after_missing: pd.Series) -> pd.DataFrame:
    report = pd.DataFrame({
        "column": before_missing.index,
        "missing_before": before_missing.values,
        "missing_after": after_missing.values,
    })
    report["filled_count"] = report["missing_before"] - report["missing_after"]
    return report


def validate_final(df: pd.DataFrame) -> None:
    # Final schema check
    missing_cols = [c for c in FINAL_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing final columns after cleaning: {missing_cols}")

    # Null check
    total_missing = int(df.isna().sum().sum())
    if total_missing > 0:
        print(f"[WARNING] Dataset still contains {total_missing} missing values after cleaning")
    else:
        print("[OK] No missing values remain after cleaning")

    # Binary checks
    for col in BINARY_COLUMNS:
        unique_vals = sorted(df[col].dropna().unique().tolist())
        if not set(unique_vals).issubset({0, 1}):
            raise ValueError(f"{col} contains invalid binary values: {unique_vals}")

    # Risk bounds
    for col in ["MerchantRisk", "CardRisk"]:
        if ((df[col] < 0) | (df[col] > 1)).any():
            raise ValueError(f"{col} contains values outside [0, 1]")

    # Amount positive
    if (df["Amount"] <= 0).any():
        raise ValueError("Amount contains non-positive values")


def main():
    parser = argparse.ArgumentParser(description="Clean merged fraud master dataset")
    parser.add_argument("--infile", required=True, help="Path to merged CSV")
    parser.add_argument("--outdir", default="/data", help="Output folder")
    parser.add_argument("--outfile", default="nigeria_credit_card_master_clean.csv", help="Clean output filename")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    infile = Path(args.infile)
    if not infile.exists():
        raise FileNotFoundError(f"Input file not found: {infile}")

    print(f"[INFO] Loading: {infile}")
    df = pd.read_csv(infile)

    print(f"[INFO] Original shape: {df.shape}")
    df = standardize_columns(df)
    df = ensure_final_columns(df)

    before_missing = df.isna().sum()

    # Clean pipeline
    df = parse_time_column(df)
    df = fill_categorical(df)
    df = fill_amount(df)
    df = fill_binary_columns(df)
    df = fill_risk_columns(df)
    df = remove_duplicates(df)

    # Time: keep parsed datetime if possible, otherwise fill with a safe placeholder
    if pd.api.types.is_datetime64_any_dtype(df["Time"]):
        if df["Time"].isna().any():
            fallback_time = pd.Timestamp("2024-01-01 12:00:00")
            df["Time"] = df["Time"].fillna(fallback_time)
        df["Time"] = df["Time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    else:
        df["Time"] = df["Time"].astype(str).replace(["nan", "None", "<NA>"], "2024-01-01 12:00:00")

    after_missing = df.isna().sum()
    report = create_missing_value_report(before_missing, after_missing)

    validate_final(df)

    # Save outputs
    clean_path = outdir / args.outfile
    report_path = outdir / "cleaning_report.csv"
    summary_path = outdir / "cleaning_summary.txt"

    df.to_csv(clean_path, index=False)
    report.to_csv(report_path, index=False)

    fraud_ratio = df["FraudFlag"].mean()

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Merged Dataset Cleaning Summary\n")
        f.write("=" * 80 + "\n")
        f.write(f"Input file: {infile}\n")
        f.write(f"Output file: {clean_path}\n")
        f.write(f"Final shape: {df.shape}\n")
        f.write(f"Fraud ratio: {fraud_ratio:.6f}\n")
        f.write("\nMissing Value Report:\n")
        f.write(report.to_string(index=False))
        f.write("\n")

    print(f"[OK] Clean dataset saved to: {clean_path}")
    print(f"[OK] Missing value report saved to: {report_path}")
    print(f"[OK] Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()