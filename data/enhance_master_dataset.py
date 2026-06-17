import argparse
from pathlib import Path
import random
from typing import Dict, List

import numpy as np
import pandas as pd


BASE_COLUMNS = [
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

TARGET_COLUMNS = BASE_COLUMNS + ["SubCategory"]


# Expanded daily-life mapping
SUBCATEGORY_MAP: Dict[str, List[str]] = {
    "Food": [
        "Restaurant", "FastFood", "Bakery", "Cafe", "StreetFood", "Takeout"
    ],
    "Retail": [
        "Supermarket", "MiniMart", "MarketStall", "POSPurchase", "DepartmentStore"
    ],
    "Bills": [
        "ElectricityBill", "WaterBill", "InternetSubscription", "CableTV",
        "WasteBill", "PhoneBill"
    ],
    "Travel": [
        "Flight", "Hotel", "Transport", "RideHailing", "BusTicket", "Fuel", "Toll"
    ],
    "Healthcare": [
        "Pharmacy", "Hospital", "Clinic", "LabTest", "Dental", "Optical"
    ],
    "Education": [
        "SchoolFees", "TuitionFees", "Books", "Stationery", "TrainingCourse",
        "ExamFees", "HostelFees", "ProjectMaterials"
    ],
    "Entertainment": [
        "Cinema", "Streaming", "Concert", "EventTicket", "Arcade", "NightOut"
    ],
    "Luxury": [
        "Jewelry", "DesignerWear", "LuxuryStore", "Watches", "Perfume"
    ],
    "Electronics": [
        "Phones", "Laptops", "Accessories", "Appliances", "Repairs", "GamingConsole"
    ],
    "Insurance": [
        "HealthInsurance", "VehicleInsurance", "LifeInsurance", "TravelInsurance"
    ],
    "Gaming": [
        "OnlineGaming", "Betting", "Lottery", "InAppPurchase"
    ],
    "Groceries": [
        "FreshProduce", "HouseholdSupplies", "Beverages", "DailyNeeds", "BulkPurchase"
    ],
}

# If your current file has only broad categories, we can still enrich them
CATEGORY_NORMALIZATION = {
    "Food": "Food",
    "Retail": "Retail",
    "Bills": "Bills",
    "Travel": "Travel",
    "Healthcare": "Healthcare",
    "Education": "Education",
    "Entertainment": "Entertainment",
    "Luxury": "Luxury",
    "Electronics": "Electronics",
    "Insurance": "Insurance",
    "Gaming": "Gaming",
    "Groceries": "Groceries",
}


def weighted_choice(options: List[str], weights: List[float], rng: random.Random) -> str:
    return rng.choices(options, weights=weights, k=1)[0]


def parse_time_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    try:
        df["Time"] = pd.to_datetime(df["Time"], errors="coerce", format="mixed")
    except TypeError:
        df["Time"] = pd.to_datetime(df["Time"], errors="coerce", infer_datetime_format=True)

    # Fill any remaining bad timestamps with a safe midday fallback
    fallback = pd.Timestamp("2024-01-01 12:00:00")
    df["Time"] = df["Time"].fillna(fallback)
    return df


def normalize_categories(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Category"] = df["Category"].astype(str).str.strip()

    def norm(cat: str) -> str:
        if cat in CATEGORY_NORMALIZATION:
            return CATEGORY_NORMALIZATION[cat]

        low = cat.lower()
        if "food" in low or "restaurant" in low or "bakery" in low or "cafe" in low:
            return "Food"
        if "retail" in low or "market" in low or "supermarket" in low or "shop" in low:
            return "Retail"
        if "bill" in low or "utility" in low:
            return "Bills"
        if "travel" in low or "flight" in low or "hotel" in low or "transport" in low:
            return "Travel"
        if "health" in low or "pharmacy" in low or "hospital" in low or "clinic" in low:
            return "Healthcare"
        if "school" in low or "education" in low or "tuition" in low or "book" in low:
            return "Education"
        if "entertain" in low or "cinema" in low or "stream" in low:
            return "Entertainment"
        if "luxury" in low or "jewelry" in low or "designer" in low:
            return "Luxury"
        if "electronic" in low or "phone" in low or "laptop" in low:
            return "Electronics"
        if "insurance" in low:
            return "Insurance"
        if "gaming" in low or "bet" in low or "lottery" in low:
            return "Gaming"
        if "grocery" in low:
            return "Groceries"

        return "Retail"

    df["Category"] = df["Category"].apply(norm)
    return df


def assign_subcategory(row: pd.Series, rng: random.Random) -> str:
    category = row["Category"]
    amount = float(row["Amount"])
    is_night = int(row["IsNightTransaction"])
    fraud = int(row["FraudFlag"])
    hour = row["Time"].hour if pd.notna(row["Time"]) else 12
    weekend = row["Time"].dayofweek >= 5 if pd.notna(row["Time"]) else False

    # Default options
    options = SUBCATEGORY_MAP.get(category, ["GeneralPurchase"])
    weights = [1.0] * len(options)

    # Add realistic behavioral bias
    if category == "Education":
        bias = {
            "SchoolFees": 4.0,
            "TuitionFees": 4.0,
            "Books": 2.5,
            "Stationery": 1.5,
            "TrainingCourse": 1.8,
            "ExamFees": 1.7,
            "HostelFees": 1.5,
            "ProjectMaterials": 1.3,
        }
        if amount > 50000:
            bias["SchoolFees"] = 5.0
            bias["TuitionFees"] = 5.0
            bias["HostelFees"] = 2.5
        if amount < 10000:
            bias["Books"] = 3.5
            bias["Stationery"] = 3.0

    elif category == "Food":
        bias = {
            "Restaurant": 2.5,
            "FastFood": 2.2,
            "Bakery": 1.8,
            "Cafe": 1.4,
            "StreetFood": 1.6,
            "Takeout": 1.5,
        }
        if hour < 8:
            bias["Bakery"] = 3.0
            bias["Cafe"] = 2.4
        if is_night:
            bias["FastFood"] = 3.0
            bias["Takeout"] = 2.5
            bias["Restaurant"] = 2.0

    elif category == "Bills":
        bias = {
            "ElectricityBill": 2.5,
            "WaterBill": 1.5,
            "InternetSubscription": 2.2,
            "CableTV": 1.8,
            "WasteBill": 1.0,
            "PhoneBill": 2.0,
        }

    elif category == "Travel":
        bias = {
            "Flight": 1.3,
            "Hotel": 1.5,
            "Transport": 2.2,
            "RideHailing": 2.0,
            "BusTicket": 1.7,
            "Fuel": 2.4,
            "Toll": 1.1,
        }
        if amount > 50000:
            bias["Flight"] = 3.5
            bias["Hotel"] = 2.8
        if amount < 10000:
            bias["RideHailing"] = 3.0
            bias["BusTicket"] = 2.8
            bias["Fuel"] = 3.2

    elif category == "Healthcare":
        bias = {
            "Pharmacy": 2.8,
            "Hospital": 2.0,
            "Clinic": 1.8,
            "LabTest": 1.5,
            "Dental": 1.0,
            "Optical": 1.0,
        }
        if amount > 30000:
            bias["Hospital"] = 3.0
            bias["LabTest"] = 2.0

    elif category == "Retail":
        bias = {
            "Supermarket": 2.8,
            "MiniMart": 1.9,
            "MarketStall": 1.7,
            "POSPurchase": 1.2,
            "DepartmentStore": 1.5,
        }
        if weekend:
            bias["Supermarket"] = 3.2
            bias["DepartmentStore"] = 2.1

    elif category == "Groceries":
        bias = {
            "FreshProduce": 2.4,
            "HouseholdSupplies": 2.0,
            "Beverages": 1.5,
            "DailyNeeds": 2.5,
            "BulkPurchase": 1.2,
        }
        if amount > 40000:
            bias["BulkPurchase"] = 3.0

    elif category == "Gaming":
        bias = {
            "OnlineGaming": 2.2,
            "Betting": 3.2,
            "Lottery": 1.4,
            "InAppPurchase": 1.8,
        }
        if fraud == 1:
            bias["Betting"] = 4.0
            bias["OnlineGaming"] = 2.8

    elif category == "Electronics":
        bias = {
            "Phones": 2.6,
            "Laptops": 1.8,
            "Accessories": 2.0,
            "Appliances": 1.4,
            "Repairs": 1.1,
            "GamingConsole": 1.0,
        }
        if amount > 100000:
            bias["Phones"] = 3.2
            bias["Laptops"] = 2.8
            bias["Appliances"] = 2.0

    elif category == "Entertainment":
        bias = {
            "Cinema": 1.7,
            "Streaming": 2.5,
            "Concert": 1.2,
            "EventTicket": 1.5,
            "Arcade": 1.0,
            "NightOut": 1.8,
        }
        if is_night or weekend:
            bias["NightOut"] = 2.8
            bias["Cinema"] = 2.1

    elif category == "Luxury":
        bias = {
            "Jewelry": 1.6,
            "DesignerWear": 2.0,
            "LuxuryStore": 2.2,
            "Watches": 1.3,
            "Perfume": 1.4,
        }

    elif category == "Insurance":
        bias = {
            "HealthInsurance": 2.1,
            "VehicleInsurance": 2.0,
            "LifeInsurance": 1.6,
            "TravelInsurance": 1.0,
        }

    else:
        bias = {opt: 1.0 for opt in options}

    weights = [bias.get(opt, 1.0) for opt in options]
    return weighted_choice(options, weights, rng)


def rebuild_category_for_specific_daily_life(df: pd.DataFrame, rng: random.Random) -> pd.DataFrame:
    """
    This stage injects a bit more daily-life realism by moving some records
    into Education / Bills / Food / Retail / Travel where the amount/time patterns fit.
    It does not destroy the overall dataset, it just improves realism.
    """
    df = df.copy()

    # Very small controlled remapping only where broad categories are too generic
    retail_mask = df["Category"].eq("Retail")
    idx = df[retail_mask].sample(frac=0.15, random_state=42).index if retail_mask.any() else []

    for i in idx:
        amt = float(df.at[i, "Amount"])
        hour = df.at[i, "Time"].hour

        if amt > 60000:
            df.at[i, "Category"] = rng.choice(["Education", "Electronics", "Travel"])
        elif amt < 5000 and hour < 9:
            df.at[i, "Category"] = rng.choice(["Food", "Groceries"])
        elif 5000 <= amt <= 20000:
            df.at[i, "Category"] = rng.choice(["Bills", "Food", "Groceries"])

    return df


def create_enhanced_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for category, sub in df.groupby("Category"):
        rows.append({
            "Category": category,
            "Count": len(sub),
            "FraudRate": round(sub["FraudFlag"].mean(), 6),
            "UniqueSubCategories": sub["SubCategory"].nunique(),
        })
    return pd.DataFrame(rows).sort_values("Count", ascending=False)


def main():
    parser = argparse.ArgumentParser(description="Enhance cleaned master dataset with richer daily-life categories")
    parser.add_argument("--infile", required=True, help="Path to cleaned master CSV")
    parser.add_argument("--outdir", default="data", help="Output directory")
    parser.add_argument("--outfile", default="nigeria_credit_card_master_enhanced.csv", help="Output CSV filename")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    infile = Path(args.infile)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(infile)

    missing = [c for c in BASE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Input dataset is missing required columns: {missing}")

    # Work on expected base columns only
    df = df[BASE_COLUMNS].copy()

    # Clean types
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(df["Amount"].median()).clip(lower=0.01)
    df["FraudFlag"] = pd.to_numeric(df["FraudFlag"], errors="coerce").fillna(0).astype(int)
    df["HighAmountFlag"] = pd.to_numeric(df["HighAmountFlag"], errors="coerce").fillna(0).astype(int)
    df["IsNightTransaction"] = pd.to_numeric(df["IsNightTransaction"], errors="coerce").fillna(0).astype(int)
    df["MerchantRisk"] = pd.to_numeric(df["MerchantRisk"], errors="coerce").fillna(df["FraudFlag"].mean()).clip(0, 1)
    df["CardRisk"] = pd.to_numeric(df["CardRisk"], errors="coerce").fillna(df["FraudFlag"].mean()).clip(0, 1)

    for col in ["Merchant", "Category", "CardType", "Location"]:
        df[col] = df[col].astype(str).str.strip().replace({"": "Unknown", "nan": "Unknown"})

    df = parse_time_column(df)
    df = normalize_categories(df)

    # Controlled realism enhancement
    df = rebuild_category_for_specific_daily_life(df, rng)

    # Re-normalize after controlled reassignment
    df = normalize_categories(df)

    # Add SubCategory
    df["SubCategory"] = df.apply(lambda row: assign_subcategory(row, rng), axis=1)

    # Recompute binary high amount flag based on full enhanced data
    high_threshold = df["Amount"].quantile(0.90)
    df["HighAmountFlag"] = (df["Amount"] >= high_threshold).astype(int)

    # Recompute IsNightTransaction from parsed time for consistency
    df["IsNightTransaction"] = df["Time"].dt.hour.isin([23, 0, 1, 2, 3, 4, 5]).astype(int)

    # Save Time back to string
    df["Time"] = df["Time"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Final column order
    df = df[TARGET_COLUMNS]

    out_csv = outdir / args.outfile
    out_summary = outdir / "enhanced_master_summary.csv"
    out_txt = outdir / "enhanced_master_summary.txt"

    df.to_csv(out_csv, index=False)

    summary = create_enhanced_summary(df)
    summary.to_csv(out_summary, index=False)

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("Enhanced Master Dataset Summary\n")
        f.write("=" * 80 + "\n")
        f.write(f"Input file: {infile}\n")
        f.write(f"Output file: {out_csv}\n")
        f.write(f"Rows: {len(df)}\n")
        f.write(f"Columns: {list(df.columns)}\n")
        f.write(f"Overall fraud ratio: {df['FraudFlag'].mean():.6f}\n\n")
        f.write("Category Summary:\n")
        f.write(summary.to_string(index=False))
        f.write("\n\nTop 30 SubCategories:\n")
        f.write(df["SubCategory"].value_counts().head(30).to_string())
        f.write("\n")

    print(f"[OK] Enhanced dataset saved to: {out_csv}")
    print(f"[OK] Summary CSV saved to: {out_summary}")
    print(f"[OK] Summary TXT saved to: {out_txt}")


if __name__ == "__main__":
    main()