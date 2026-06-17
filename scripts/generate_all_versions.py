from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

RANDOM_STATE = 42
FIXED_TIME_DATE = "2024-01-01"

BASE_COLUMNS = ["Amount", "Merchant", "Category", "CardType", "Location", "Time", "FraudFlag"]
STABLE_ID_COLUMNS = BASE_COLUMNS + [
    "HighAmountFlag",
    "IsNightTransaction",
    "MerchantRisk",
    "CardRisk",
]

SOURCE_FILES = [
    "data_old/nigeria_credit_card_fraud_dataset_v1.csv",
    "data_old/nigeria_credit_card_fraud_dataset_v2.csv",
    "data_old/nigeria_credit_card_fraud_dataset_v3.csv",
    "data_old/nigeria_credit_card_fraud_dataset_v4.csv",
    "data_old/nigeria_credit_card_fraud_dataset_v5.csv",
    "data/nigeria_credit_card_fraud_dataset_v1.csv",
    "data/nigeria_credit_card_fraud_dataset_v2.csv",
    "data/nigeria_credit_card_fraud_dataset_v3.csv",
    "data/nigeria_credit_card_fraud_dataset_v4.csv",
    "data/nigeria_credit_card_fraud_dataset_v5.csv",
    "data/nigeria_credit_card_merged_v3_v4_v5.csv",
    "data/nigeria_credit_card_master_clean.csv",
    "data/nigeria_credit_card_master_enhanced.csv",
    "data/dataset_v1.csv",
    "data/dataset_v2.csv",
    "data/dataset_v3.csv",
    "data/dataset_v4.csv",
    "data/dataset_v5.csv",
]

VERSION_COLUMNS = {
    "v1": ["TransactionID"] + BASE_COLUMNS,
    "v2": ["TransactionID"] + BASE_COLUMNS + ["HighAmountFlag", "IsNightTransaction", "SubCategory"],
    "v3": [
        "TransactionID",
        *BASE_COLUMNS,
        "HighAmountFlag",
        "IsNightTransaction",
        "SubCategory",
        "MerchantRisk",
        "CardRisk",
    ],
    "v4": [
        "TransactionID",
        *BASE_COLUMNS,
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
    "v5": [
        "TransactionID",
        *BASE_COLUMNS,
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

SUBCATEGORY_MAP = {
    "Food": ["Restaurant", "FastFood", "Bakery", "Cafe", "StreetFood", "Takeout"],
    "Retail": ["Supermarket", "MiniMart", "MarketStall", "POSPurchase", "DepartmentStore"],
    "Bills": ["ElectricityBill", "WaterBill", "InternetSubscription", "CableTV", "WasteBill", "PhoneBill"],
    "Travel": ["Flight", "Hotel", "Transport", "RideHailing", "BusTicket", "Fuel", "Toll"],
    "Healthcare": ["Pharmacy", "Hospital", "Clinic", "LabTest", "Dental", "Optical"],
    "Education": [
        "SchoolFees",
        "TuitionFees",
        "Books",
        "Stationery",
        "TrainingCourse",
        "ExamFees",
        "HostelFees",
        "ProjectMaterials",
    ],
    "Entertainment": ["Cinema", "Streaming", "Concert", "EventTicket", "Arcade", "NightOut"],
    "Luxury": ["Jewelry", "DesignerWear", "LuxuryStore", "Watches", "Perfume"],
    "Electronics": ["Phones", "Laptops", "Accessories", "Appliances", "Repairs", "GamingConsole"],
    "Insurance": ["HealthInsurance", "VehicleInsurance", "LifeInsurance", "TravelInsurance"],
    "Gaming": ["OnlineGaming", "Betting", "Lottery", "InAppPurchase"],
    "Groceries": ["FreshProduce", "HouseholdSupplies", "Beverages", "DailyNeeds", "BulkPurchase"],
}


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    for col in df.columns:
        clean = col.strip()
        low = clean.lower()
        if clean in ["Amount_₦", "Amount_â‚¦", "Amount(NGN)", "TransactionAmount"] or low == "amount":
            rename_map[col] = "Amount"
        elif low in ["merchant", "merchanttype"]:
            rename_map[col] = "Merchant"
        elif low == "category":
            rename_map[col] = "Category"
        elif low == "cardtype":
            rename_map[col] = "CardType"
        elif low == "location":
            rename_map[col] = "Location"
        elif low == "time":
            rename_map[col] = "Time"
        elif low in ["fraudflag", "isfraud", "fraud"]:
            rename_map[col] = "FraudFlag"
        elif low == "highamountflag":
            rename_map[col] = "HighAmountFlag"
        elif low == "isnighttransaction":
            rename_map[col] = "IsNightTransaction"
        elif low == "merchantrisk":
            rename_map[col] = "MerchantRisk"
        elif low == "cardrisk":
            rename_map[col] = "CardRisk"

    df = df.rename(columns=rename_map)
    df.columns = [c.strip() for c in df.columns]
    return df


def normalize_category(value: Any) -> str:
    text = str(value).strip()
    low = text.lower()
    direct = {k.lower(): k for k in SUBCATEGORY_MAP}
    if low in direct:
        return direct[low]
    if "food" in low or "restaurant" in low or "cafe" in low:
        return "Food"
    if "retail" in low or "shop" in low or "market" in low or "mall" in low or low in ["jumia", "konga"]:
        return "Retail"
    if "bill" in low or "utility" in low:
        return "Bills"
    if "travel" in low or "flight" in low or "hotel" in low or "fuel" in low:
        return "Travel"
    if "health" in low or "pharma" in low or "hospital" in low:
        return "Healthcare"
    if "education" in low or "school" in low or "tuition" in low:
        return "Education"
    if "entertain" in low or "cinema" in low or "stream" in low:
        return "Entertainment"
    if "luxury" in low or "jewelry" in low or "designer" in low:
        return "Luxury"
    if "electro" in low or "phone" in low or "laptop" in low:
        return "Electronics"
    if "insurance" in low:
        return "Insurance"
    if "gaming" in low or "bet" in low or "lottery" in low:
        return "Gaming"
    if "grocery" in low or "groceries" in low:
        return "Groceries"
    return "Retail"


def normalize_card_type(value: Any) -> str:
    text = str(value).strip()
    low = text.lower().replace(" ", "")
    mapping = {
        "visa": "Visa",
        "mastercard": "MasterCard",
        "verve": "Verve",
        "amex": "Amex",
        "virtualcard": "VirtualCard",
    }
    return mapping.get(low, text if text else "Unknown")


def normalize_location(value: Any) -> str:
    text = str(value).strip()
    low = text.lower().replace(" ", "")
    mapping = {
        "portharcourt": "Port Harcourt",
        "lagos": "Lagos",
        "abuja": "Abuja",
        "kano": "Kano",
        "ibadan": "Ibadan",
        "jos": "Jos",
        "uyo": "Uyo",
    }
    return mapping.get(low, text.title() if text else "Unknown")


def parse_time_value(value: Any) -> pd.Timestamp:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return pd.Timestamp(f"{FIXED_TIME_DATE} 12:00:00")

    time_only = pd.to_datetime(text, format="%H:%M", errors="coerce")
    if pd.notna(time_only):
        return pd.Timestamp(f"{FIXED_TIME_DATE} {time_only.strftime('%H:%M:%S')}")

    parsed = pd.to_datetime(text, errors="coerce", format="mixed")
    if pd.isna(parsed):
        return pd.Timestamp(f"{FIXED_TIME_DATE} 12:00:00")
    return pd.Timestamp(parsed).floor("s")


def parse_time_series(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    missing = text.isna() | text.str.lower().isin(["", "nan", "none", "<na>"])
    time_only = text.str.match(r"^\d{1,2}:\d{2}(:\d{2})?$", na=False)

    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    if time_only.any():
        parsed.loc[time_only] = pd.to_datetime(
            FIXED_TIME_DATE + " " + text.loc[time_only],
            errors="coerce",
            format="mixed",
        )
    date_like = ~(missing | time_only)
    if date_like.any():
        parsed.loc[date_like] = pd.to_datetime(text.loc[date_like], errors="coerce", format="mixed")

    fallback = pd.Timestamp(f"{FIXED_TIME_DATE} 12:00:00")
    return parsed.fillna(fallback).dt.floor("s")


def normalize_risk(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() == 0:
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)
    values = values.fillna(values.median())
    max_val = values.max()
    min_val = values.min()
    if max_val > 1 or min_val < 0:
        denom = max_val - min_val
        values = (values - min_val) / denom if denom else values * 0
    return values.clip(0, 1).astype(float)


def canonical_value(value: Any) -> str:
    if pd.isna(value):
        return "<NA>"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value).strip()


def stable_payload_series(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    parts: list[pd.Series] = []
    for col in columns:
        if col == "Amount":
            part = pd.to_numeric(df[col], errors="coerce").round(2).astype("string").fillna("<NA>")
        elif col in ["FraudFlag", "HighAmountFlag", "IsNightTransaction"]:
            part = pd.to_numeric(df[col], errors="coerce").round().astype("Int64").astype("string").fillna("<NA>")
        elif col in ["MerchantRisk", "CardRisk"]:
            part = pd.to_numeric(df[col], errors="coerce").round(6).astype("string").fillna("<NA>")
        else:
            part = df[col].astype("string").str.strip().fillna("<NA>")
        parts.append(part)

    payload = parts[0]
    for part in parts[1:]:
        payload = payload + "|" + part
    return payload


def hash_payloads(payloads: pd.Series, length: int = 24) -> pd.Series:
    return payloads.map(lambda value: hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:length])


def assign_subcategories(df: pd.DataFrame) -> pd.Series:
    payloads = stable_payload_series(df, ["Merchant", "Category", "Amount", "Time"])
    hashes = hash_payloads(payloads, length=8)
    values = []
    for category, hex_hash in zip(df["Category"], hashes):
        options = SUBCATEGORY_MAP.get(category, ["GeneralPurchase"])
        idx = int(hex_hash, 16) % len(options)
        values.append(options[idx])
    return pd.Series(values, index=df.index)


def clean_master_from_merged(merged_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = standardize_columns(pd.read_csv(merged_path))
    missing = [c for c in BASE_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"{merged_path} is missing required base columns: {missing}")

    df = raw.copy()
    before_rows = len(df)
    exact_duplicates_before = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)

    for col in ["HighAmountFlag", "IsNightTransaction", "MerchantRisk", "CardRisk"]:
        if col not in df.columns:
            df[col] = np.nan

    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    amount_median = df["Amount"].median()
    df["Amount"] = df["Amount"].fillna(amount_median).clip(lower=0.01).round(2)
    df["FraudFlag"] = pd.to_numeric(df["FraudFlag"], errors="coerce").fillna(0)
    df["FraudFlag"] = df["FraudFlag"].apply(lambda x: 1 if x >= 0.5 else 0).astype(int)

    df["Merchant"] = df["Merchant"].astype("string").str.strip().fillna("Unknown")
    df["Merchant"] = df["Merchant"].replace({"": "Unknown", "nan": "Unknown", "None": "Unknown"})
    df["Category"] = df["Category"].apply(normalize_category)
    df["CardType"] = df["CardType"].apply(normalize_card_type)
    df["Location"] = df["Location"].apply(normalize_location)
    df["Time"] = parse_time_series(df["Time"])

    df["HighAmountFlag"] = (df["Amount"] >= df["Amount"].quantile(0.90)).astype(int)
    df["IsNightTransaction"] = df["Time"].dt.hour.isin([23, 0, 1, 2, 3, 4, 5]).astype(int)
    df["MerchantRisk"] = normalize_risk(df["MerchantRisk"])
    df["CardRisk"] = normalize_risk(df["CardRisk"])
    df["Time"] = df["Time"].dt.strftime("%Y-%m-%d %H:%M:%S")

    df["TransactionID"] = hash_payloads(stable_payload_series(df, STABLE_ID_COLUMNS))
    duplicate_transaction_ids_before = int(df["TransactionID"].duplicated().sum())
    df = df.drop_duplicates(subset=["TransactionID"], keep="first").reset_index(drop=True)

    near_cols = ["Amount", "Merchant", "Category", "CardType", "Location", "Time"]
    near_duplicate_count = int(df.duplicated(subset=near_cols, keep=False).sum())
    near_duplicate_groups = int(df.loc[df.duplicated(subset=near_cols, keep=False), near_cols].drop_duplicates().shape[0])

    add_engineered_features(df)

    report = {
        "input_rows": before_rows,
        "exact_duplicate_rows_removed": exact_duplicates_before,
        "duplicate_transaction_identities_in_merged": exact_duplicates_before + duplicate_transaction_ids_before,
        "duplicate_transaction_ids_removed": duplicate_transaction_ids_before,
        "final_rows": len(df),
        "near_duplicate_records_not_removed": near_duplicate_count,
        "near_duplicate_groups_not_removed": near_duplicate_groups,
    }
    return df, report


def add_engineered_features(df: pd.DataFrame) -> None:
    dt = pd.to_datetime(df["Time"], errors="coerce", format="mixed")
    df["Hour"] = dt.dt.hour.fillna(12).astype(int)
    df["DayOfWeek"] = dt.dt.dayofweek.fillna(0).astype(int)
    df["Month"] = dt.dt.month.fillna(1).astype(int)
    df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)
    df["SubCategory"] = assign_subcategories(df)
    df["AmountLog"] = np.log1p(df["Amount"]).round(6)
    category_median = df.groupby("Category")["Amount"].transform("median").replace(0, np.nan)
    df["AmountToCategoryMedian"] = (df["Amount"] / category_median).replace([np.inf, -np.inf], np.nan).fillna(1).round(6)
    category_mean = df.groupby("Category")["Amount"].transform("mean")
    category_std = df.groupby("Category")["Amount"].transform("std").replace(0, np.nan)
    df["AmountZScoreByCategory"] = ((df["Amount"] - category_mean) / category_std).fillna(0).round(6)
    df["AmountPercentile"] = df["Amount"].rank(method="average", pct=True).round(6)
    df["NightHighAmountFlag"] = ((df["IsNightTransaction"] == 1) & (df["HighAmountFlag"] == 1)).astype(int)
    df["RiskScoreComposite"] = (
        0.30 * df["MerchantRisk"]
        + 0.30 * df["CardRisk"]
        + 0.25 * df["HighAmountFlag"]
        + 0.15 * df["IsNightTransaction"]
    ).round(6)


def audit_file(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {"source_file": str(path), "exists": path.exists()}
    if not path.exists():
        return record

    df = standardize_columns(pd.read_csv(path))
    record.update(
        rows=len(df),
        columns=len(df.columns),
        feature_list="|".join(df.columns),
        missing_total=int(df.isna().sum().sum()),
        exact_duplicate_rows=int(df.duplicated().sum()),
    )
    if "FraudFlag" in df.columns:
        y = pd.to_numeric(df["FraudFlag"], errors="coerce")
        record.update(
            fraud_count=int((y == 1).sum()),
            legitimate_count=int((y == 0).sum()),
            fraud_ratio=round(float((y == 1).mean()), 6),
            invalid_fraudflag_count=int((~y.isin([0, 1])).sum()),
        )
    if "Amount" in df.columns:
        amount = pd.to_numeric(df["Amount"], errors="coerce")
        record.update(
            missing_amount_count=int(amount.isna().sum()),
            nonpositive_amount_count=int((amount <= 0).sum()),
            unrealistic_amount_count=int((amount > 10_000_000).sum()),
            amount_min=float(amount.min()) if amount.notna().any() else np.nan,
            amount_max=float(amount.max()) if amount.notna().any() else np.nan,
        )
    if "Time" in df.columns:
        parsed = parse_time_series(df["Time"])
        original_missing = df["Time"].astype("string").str.strip().str.lower().isin(["", "nan", "none", "<na>"]).sum()
        record.update(invalid_time_count=int(original_missing), time_min=str(parsed.min()), time_max=str(parsed.max()))
    for col in ["Merchant", "Category", "CardType", "Location", "SubCategory"]:
        if col in df.columns:
            values = df[col].astype("string").str.strip()
            bad = values.isna() | values.str.lower().isin(["", "unknown", "nan", "none", "<na>"])
            record[f"{col}_unique"] = int(values.nunique(dropna=True))
            record[f"{col}_blank_or_unknown"] = int(bad.sum())
    return record


def archive_existing_outputs(data_dir: Path, archive_root: Path) -> Path:
    archive_dir = archive_root / f"dataset_rebuild_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    archive_dir.mkdir(parents=True, exist_ok=True)

    for filename in [
        "dataset_v1.csv",
        "dataset_v2.csv",
        "dataset_v3.csv",
        "dataset_v4.csv",
        "dataset_v5.csv",
        "dataset_summary.csv",
        "research_master_dataset.csv",
    ]:
        source = data_dir / filename
        if source.exists():
            shutil.copy2(source, archive_dir / filename)

    merged = data_dir / "nigeria_credit_card_merged_v3_v4_v5.csv"
    appendix_dir = archive_root / "appendix_only"
    appendix_dir.mkdir(parents=True, exist_ok=True)
    if merged.exists():
        shutil.copy2(merged, appendix_dir / merged.name)
        readme = appendix_dir / "README.md"
        readme.write_text(
            "The merged v3/v4/v5 file is retained for provenance and appendix-only audit use. "
            "It is not valid as a sixth main training experiment because it is a stacked file "
            "with repeated transaction identities.\n",
            encoding="utf-8",
        )

    return archive_dir


def source_overlap_summary() -> dict[str, Any]:
    files = {
        "original_v3": Path("data/nigeria_credit_card_fraud_dataset_v3.csv"),
        "original_v4": Path("data/nigeria_credit_card_fraud_dataset_v4.csv"),
        "original_v5": Path("data/nigeria_credit_card_fraud_dataset_v5.csv"),
    }
    if not all(path.exists() for path in files.values()):
        return {"overlap_available": False}

    sets: dict[str, set[str]] = {}
    for name, path in files.items():
        df = standardize_columns(pd.read_csv(path))
        for col in ["HighAmountFlag", "IsNightTransaction", "MerchantRisk", "CardRisk"]:
            if col not in df.columns:
                df[col] = np.nan
        df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").round(2)
        df["FraudFlag"] = pd.to_numeric(df["FraudFlag"], errors="coerce").fillna(0).astype(int)
        df["Time"] = parse_time_series(df["Time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        df["MerchantRisk"] = normalize_risk(df["MerchantRisk"])
        df["CardRisk"] = normalize_risk(df["CardRisk"])
        df["HighAmountFlag"] = pd.to_numeric(df["HighAmountFlag"], errors="coerce").fillna(0).astype(int)
        df["IsNightTransaction"] = pd.to_numeric(df["IsNightTransaction"], errors="coerce").fillna(0).astype(int)
        df["Merchant"] = df["Merchant"].astype("string").str.strip().fillna("Unknown")
        df["Category"] = df["Category"].apply(normalize_category)
        df["CardType"] = df["CardType"].apply(normalize_card_type)
        df["Location"] = df["Location"].apply(normalize_location)
        ids = hash_payloads(stable_payload_series(df, STABLE_ID_COLUMNS))
        sets[name] = set(ids)

    return {
        "overlap_available": True,
        "original_v3_unique_ids": len(sets["original_v3"]),
        "original_v4_unique_ids": len(sets["original_v4"]),
        "original_v5_unique_ids": len(sets["original_v5"]),
        "v3_v4_overlap": len(sets["original_v3"] & sets["original_v4"]),
        "v3_v5_overlap": len(sets["original_v3"] & sets["original_v5"]),
        "v4_v5_overlap": len(sets["original_v4"] & sets["original_v5"]),
    }


def write_outputs(master: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    master_columns = list(dict.fromkeys(VERSION_COLUMNS["v5"]))
    master[master_columns].to_csv(data_dir / "research_master_dataset.csv", index=False)

    rows = []
    for version, columns in VERSION_COLUMNS.items():
        out = master[columns].copy()
        out.to_csv(data_dir / f"dataset_{version}.csv", index=False)
        rows.append(
            {
                "version": version,
                "rows": len(out),
                "columns": len(out.columns),
                "fraud_count": int(out["FraudFlag"].sum()),
                "legitimate_count": int((out["FraudFlag"] == 0).sum()),
                "fraud_ratio": round(float(out["FraudFlag"].mean()), 6),
                "features": "|".join(out.columns),
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(data_dir / "dataset_summary.csv", index=False)
    return summary


def write_reports(
    reports_dir: Path,
    source_audit: pd.DataFrame,
    version_summary: pd.DataFrame,
    clean_report: dict[str, Any],
    overlap: dict[str, Any],
    archive_dir: Path,
) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = pd.concat(
        [
            source_audit,
            version_summary.assign(source_file=lambda d: "final_" + d["version"], exists=True),
        ],
        ignore_index=True,
        sort=False,
    )
    summary_rows.to_csv(reports_dir / "dataset_audit_summary.csv", index=False)

    lines = [
        "# Dataset Audit Report",
        "",
        "## Source Files Inspected",
        "",
    ]
    for _, row in source_audit.iterrows():
        status = "present" if bool(row.get("exists")) else "missing"
        lines.append(f"- `{row['source_file']}`: {status}")

    lines.extend(
        [
            "",
            "## Current Generation Lineage",
            "",
            "- `data/genrate_all_versions.py` is a legacy typo-named generator that resamples data and can upsample with jitter.",
            "- `scripts/generate_v5.py` creates a size/ratio-controlled v5 file by resampling and can duplicate rows when upsampling.",
            "- `data/clean_merged_master.py` cleaned `nigeria_credit_card_merged_v3_v4_v5.csv` and removed duplicate rows/transaction identities.",
            "- `data/enhance_master_dataset.py` added daily-life category/subcategory enrichment.",
            "- `data/create_stratified_versions.py` created previous `dataset_v1.csv` through `dataset_v5.csv` as increasing-size stratified samples from the enhanced master.",
            "- New `scripts/generate_all_versions.py` rebuilds all final training datasets from one unique master population with deterministic transaction IDs.",
            "",
            "## Duplicate And Master-Dataset Findings",
            "",
            f"- Merged input rows: {clean_report['input_rows']}",
            f"- Exact duplicate rows removed from merged input: {clean_report['exact_duplicate_rows_removed']}",
            f"- Duplicate transaction identities detected in merged input: {clean_report['duplicate_transaction_identities_in_merged']}",
            f"- Additional duplicate synthetic transaction IDs removed after exact dedupe: {clean_report['duplicate_transaction_ids_removed']}",
            f"- Final clean master rows: {clean_report['final_rows']}",
            f"- Near-duplicate records retained for review: {clean_report['near_duplicate_records_not_removed']}",
            f"- Near-duplicate groups retained for review: {clean_report['near_duplicate_groups_not_removed']}",
            "",
            "## Merged File Validity",
            "",
            "`data/nigeria_credit_card_merged_v3_v4_v5.csv` is invalid as a sixth main training experiment. "
            "It is a stacked v3/v4/v5-style file with repeated transaction identities. "
            "It is retained only for provenance and appendix-use audit.",
        ]
    )

    if overlap.get("overlap_available"):
        lines.extend(
            [
                "",
                "### v3/v4/v5 Source Overlap",
                "",
                f"- Original v3 unique IDs: {overlap['original_v3_unique_ids']}",
                f"- Original v4 unique IDs: {overlap['original_v4_unique_ids']}",
                f"- Original v5 unique IDs: {overlap['original_v5_unique_ids']}",
                f"- v3-v4 overlap: {overlap['v3_v4_overlap']}",
                f"- v3-v5 overlap: {overlap['v3_v5_overlap']}",
                f"- v4-v5 overlap: {overlap['v4_v5_overlap']}",
            ]
        )

    lines.extend(
        [
            "",
            "## Final Dataset Versions",
            "",
        ]
    )
    for _, row in version_summary.iterrows():
        lines.append(
            f"- `{row['version']}`: {row['rows']} rows, {row['columns']} columns, "
            f"fraud ratio {row['fraud_ratio']}; features: {row['features']}"
        )

    lines.extend(
        [
            "",
            "## Data Quality Notes",
            "",
            "- All final datasets are built from the same unique clean master population to avoid row-count confounding.",
            "- `TransactionID` is deterministic and derived from stable transaction fields using SHA-256 hashing.",
            "- Exact duplicate rows and duplicate transaction IDs were removed from the master.",
            "- Near-duplicates were reported but not silently deleted.",
            "- Time-only values are parsed against a fixed date (`2024-01-01`) to avoid run-date-dependent timestamps.",
            "- Categorical text fields are stripped and normalized; blank/unknown categorical values are counted in the CSV audit summary.",
            "- Legacy `MerchantRisk` and `CardRisk` are retained only in v3+ as documented synthetic risk indicators; they should be discussed as leakage-risk features in reporting.",
            "",
            "## Final Recommendation",
            "",
            "Use `data/dataset_v1.csv` through `data/dataset_v5.csv` for main training. "
            "Do not use `data/nigeria_credit_card_merged_v3_v4_v5.csv` as a separate main experiment. "
            "Keep the merged file for appendix/provenance only.",
            "",
            f"Archived replaced dataset outputs at: `{archive_dir}`",
            "",
        ]
    )
    (reports_dir / "dataset_audit_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and rebuild research-grade fraud datasets.")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--reports_dir", default="reports")
    parser.add_argument("--archive_dir", default="data/archive")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    reports_dir = Path(args.reports_dir)
    archive_root = Path(args.archive_dir)
    merged_path = data_dir / "nigeria_credit_card_merged_v3_v4_v5.csv"
    if not merged_path.exists():
        raise FileNotFoundError(f"Required merged source file not found: {merged_path}")

    source_audit = pd.DataFrame([audit_file(Path(p)) for p in SOURCE_FILES])
    archive_dir = archive_existing_outputs(data_dir, archive_root)
    master, clean_report = clean_master_from_merged(merged_path)
    version_summary = write_outputs(master, data_dir)
    overlap = source_overlap_summary()
    write_reports(reports_dir, source_audit, version_summary, clean_report, overlap, archive_dir)

    print(f"[OK] Rebuilt research master and dataset_v1-v5 in {data_dir}")
    print(f"[OK] Archived replaced outputs in {archive_dir}")
    print(f"[OK] Wrote audit report to {reports_dir / 'dataset_audit_report.md'}")
    print(f"[OK] Wrote audit summary to {reports_dir / 'dataset_audit_summary.csv'}")
    print("[INFO] Merged v3/v4/v5 file is appendix/provenance only, not a main training dataset.")


if __name__ == "__main__":
    main()
