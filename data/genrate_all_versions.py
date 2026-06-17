import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.utils import resample


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

DEFAULT_CONFIG = {
    "v1": {"rows": 50000, "fraud_ratio": 0.10},
    "v2": {"rows": 100000, "fraud_ratio": 0.08},
    "v3": {"rows": 200000, "fraud_ratio": 0.08},
    "v4": {"rows": 250000, "fraud_ratio": 0.10},
    "v5": {"rows": 350000, "fraud_ratio": 0.10},
}


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


def ensure_required_base_columns(df: pd.DataFrame) -> None:
    required = ["Amount", "Merchant", "Category", "CardType", "Location", "Time", "FraudFlag"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Master dataset is missing required base columns: {missing}. "
            f"It must contain at least {required}."
        )


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    df["FraudFlag"] = pd.to_numeric(df["FraudFlag"], errors="coerce").fillna(0).astype(int)

    # Normalize binary label just in case
    df["FraudFlag"] = df["FraudFlag"].apply(lambda x: 1 if x == 1 else 0)

    # Parse time
    parsed_time = pd.to_datetime(df["Time"], errors="coerce")
    if parsed_time.notna().sum() > 0:
        df["Time"] = parsed_time
    else:
        # Keep original strings if timestamps are not parseable
        df["Time"] = df["Time"].astype(str)

    # String columns
    for col in ["Merchant", "Category", "CardType", "Location"]:
        df[col] = df[col].astype(str).fillna("Unknown")

    return df


def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # HighAmountFlag
    if "HighAmountFlag" not in df.columns:
        threshold = df["Amount"].quantile(0.90)
        df["HighAmountFlag"] = (df["Amount"] >= threshold).astype(int)

    # IsNightTransaction
    if "IsNightTransaction" not in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df["Time"]):
            hours = df["Time"].dt.hour.fillna(12).astype(int)
            df["IsNightTransaction"] = hours.isin([0, 1, 2, 3, 4, 5, 23]).astype(int)
        else:
            # fallback if Time could not be parsed
            df["IsNightTransaction"] = 0

    # MerchantRisk: fraud rate by merchant
    if "MerchantRisk" not in df.columns:
        merchant_risk = df.groupby("Merchant")["FraudFlag"].mean().to_dict()
        df["MerchantRisk"] = df["Merchant"].map(merchant_risk).fillna(df["FraudFlag"].mean())

    # CardRisk: fraud rate by card type
    if "CardRisk" not in df.columns:
        card_risk = df.groupby("CardType")["FraudFlag"].mean().to_dict()
        df["CardRisk"] = df["CardType"].map(card_risk).fillna(df["FraudFlag"].mean())

    # Clean engineered types
    df["HighAmountFlag"] = pd.to_numeric(df["HighAmountFlag"], errors="coerce").fillna(0).astype(int)
    df["IsNightTransaction"] = pd.to_numeric(df["IsNightTransaction"], errors="coerce").fillna(0).astype(int)
    df["MerchantRisk"] = pd.to_numeric(df["MerchantRisk"], errors="coerce").fillna(df["FraudFlag"].mean())
    df["CardRisk"] = pd.to_numeric(df["CardRisk"], errors="coerce").fillna(df["FraudFlag"].mean())

    return df


def finalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in FINAL_COLUMNS:
        if col not in df.columns:
            if col in ["HighAmountFlag", "IsNightTransaction"]:
                df[col] = 0
            elif col in ["MerchantRisk", "CardRisk"]:
                df[col] = df["FraudFlag"].mean()
            else:
                df[col] = ""

    return df[FINAL_COLUMNS]


def sanitize_resampled_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["FraudFlag"] = df["FraudFlag"].round().clip(0, 1).astype(int)
    df["HighAmountFlag"] = df["HighAmountFlag"].round().clip(0, 1).astype(int)
    df["IsNightTransaction"] = df["IsNightTransaction"].round().clip(0, 1).astype(int)

    # Risks are probabilities
    df["MerchantRisk"] = df["MerchantRisk"].clip(0, 1)
    df["CardRisk"] = df["CardRisk"].clip(0, 1)

    # Amount must remain positive
    df["Amount"] = df["Amount"].clip(lower=0.01)

    return df


def add_small_numeric_jitter(
    df: pd.DataFrame,
    seed: int,
    exclude_cols: List[str] = None,
    noise_scale: float = 0.002,
) -> pd.DataFrame:
    df = df.copy()
    exclude_cols = exclude_cols or []

    rng = np.random.default_rng(seed)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]

    for col in numeric_cols:
        std = df[col].std()
        if pd.isna(std) or std == 0:
            continue
        noise = rng.normal(loc=0.0, scale=std * noise_scale, size=len(df))
        df[col] = df[col] + noise

    return df


def sample_class_subset(
    df_class: pd.DataFrame,
    n_samples: int,
    seed: int,
    allow_jitter: bool = True,
) -> pd.DataFrame:
    if len(df_class) >= n_samples:
        return resample(df_class, replace=False, n_samples=n_samples, random_state=seed)

    sampled = resample(df_class, replace=True, n_samples=n_samples, random_state=seed)

    # Only jitter continuous numeric columns, never labels or binary flags
    if allow_jitter:
        sampled = add_small_numeric_jitter(
            sampled,
            seed=seed,
            exclude_cols=["FraudFlag", "HighAmountFlag", "IsNightTransaction"],
            noise_scale=0.002,
        )

    return sanitize_resampled_data(sampled)


def generate_version(
    df: pd.DataFrame,
    total_rows: int,
    fraud_ratio: float,
    seed: int,
) -> pd.DataFrame:
    desired_fraud = int(round(total_rows * fraud_ratio))
    desired_legit = total_rows - desired_fraud

    fraud = df[df["FraudFlag"] == 1]
    legit = df[df["FraudFlag"] == 0]

    if fraud.empty:
        raise ValueError("No fraud rows found in master dataset.")
    if legit.empty:
        raise ValueError("No legitimate rows found in master dataset.")

    fraud_new = sample_class_subset(fraud, desired_fraud, seed=seed + 1, allow_jitter=True)
    legit_new = sample_class_subset(legit, desired_legit, seed=seed + 2, allow_jitter=False)

    out_df = pd.concat([fraud_new, legit_new], axis=0)
    out_df = out_df.sample(frac=1, random_state=seed + 3).reset_index(drop=True)
    out_df = sanitize_resampled_data(out_df)

    return out_df


def summarize_dataset(df: pd.DataFrame, version: str) -> Dict:
    fraud_count = int(df["FraudFlag"].sum())
    total = len(df)
    legit_count = total - fraud_count

    return {
        "version": version,
        "rows": total,
        "fraud_count": fraud_count,
        "legit_count": legit_count,
        "fraud_ratio": round(float(fraud_count / total), 6),
        "columns": df.columns.tolist(),
    }


def load_config(config_path: str = None) -> Dict[str, Dict]:
    if not config_path:
        return DEFAULT_CONFIG
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate v1-v5 fraud datasets with one consistent schema.")
    parser.add_argument("--infile", required=True, help="Path to master input CSV")
    parser.add_argument("--outdir", default="/data", help="Output directory, default=/data")
    parser.add_argument("--config", default=None, help="Optional JSON config for versions")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)

    df = pd.read_csv(args.infile)
    df = standardize_columns(df)
    ensure_required_base_columns(df)
    df = coerce_types(df)
    df = derive_features(df)
    df = finalize_schema(df)

    # Drop rows with bad critical values
    df = df.dropna(subset=["Amount", "Merchant", "Category", "CardType", "Location", "FraudFlag"]).reset_index(drop=True)

    summaries = []

    # Save cleaned master too
    master_out = outdir / "nigeria_credit_card_fraud_master_clean.csv"
    df.to_csv(master_out, index=False)

    for idx, (version, spec) in enumerate(config.items(), start=1):
        rows = int(spec["rows"])
        fraud_ratio = float(spec["fraud_ratio"])

        version_df = generate_version(
            df=df,
            total_rows=rows,
            fraud_ratio=fraud_ratio,
            seed=args.seed + idx * 10,
        )

        outfile = outdir / f"nigeria_credit_card_fraud_dataset_{version}.csv"
        version_df.to_csv(outfile, index=False)

        summaries.append(summarize_dataset(version_df, version))
        print(f"[OK] Saved {outfile} | rows={len(version_df)} | fraud_ratio={version_df['FraudFlag'].mean():.4f}")

    summary_df = pd.DataFrame(summaries)
    summary_csv = outdir / "dataset_generation_summary.csv"
    summary_txt = outdir / "dataset_generation_summary.txt"

    summary_df.to_csv(summary_csv, index=False)

    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("Dataset Generation Summary\n")
        f.write("=" * 80 + "\n")
        f.write(f"Master input: {args.infile}\n")
        f.write(f"Clean master saved to: {master_out}\n")
        f.write(f"Output directory: {outdir}\n\n")
        f.write(summary_df.to_string(index=False))
        f.write("\n")

    print(f"\n[OK] Summary saved to {summary_csv}")
    print(f"[OK] Summary saved to {summary_txt}")


if __name__ == "__main__":
    main()