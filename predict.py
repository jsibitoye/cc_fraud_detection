import os
import glob
import joblib
import pandas as pd
import argparse

def load_latest_model(models_root="models", version=None):
    """
    Loads the latest model pipeline.
    - If version is specified, looks in models_<version>.
    - Else, picks the most recently modified models_* directory.
    """
    if version:
        models_dir = f"models_{version}"
        if not os.path.exists(models_dir):
            raise FileNotFoundError(f"No models found for version {version}")
    else:
        candidates = sorted(
            glob.glob("models_*"),
            key=os.path.getmtime,
            reverse=True
        )
        if not candidates:
            raise FileNotFoundError("No models_* directories found.")
        models_dir = candidates[0]

    print(f"[INFO] Using models from: {models_dir}")

    joblibs = sorted(
        glob.glob(os.path.join(models_dir, "*.joblib")),
        key=os.path.getmtime,
        reverse=True
    )
    if not joblibs:
        raise FileNotFoundError(f"No .joblib files in {models_dir}")

    latest_model_path = joblibs[0]
    print(f"[INFO] Loading model: {latest_model_path}")
    model = joblib.load(latest_model_path)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=False, help="CSV file with transactions")
    parser.add_argument("--version", type=str, required=False, help="Dataset version (v1, v2, v3)")
    args = parser.parse_args()

    model = load_latest_model(version=args.version)

    # Example new transaction if no dataset is passed
    if not args.data:
        new_data = pd.DataFrame([{
            "Amount": 200,
            "Merchant": "ShopX",
            "Category": "Electronics",
            "CardType": "Visa",
            "Location": "Lagos",
            "Time": "12:30",
            "HighAmountFlag": 0,
            "IsNightTransaction": 0,
            "MerchantRisk": 1,
            "CardRisk": 1
        }])
        df = new_data
    else:
        df = pd.read_csv(args.data)

    preds = model.predict(df)
    probs = model.predict_proba(df)[:, 1]

    df["PredictedFraud"] = preds
    df["FraudProbability"] = probs

    print("\n🔎 Sample predictions:")
    print(df.head(10)[["PredictedFraud", "FraudProbability"]])

    if df["PredictedFraud"].sum() == 0:
        print("\n⚠️ WARNING: Model predicted ZERO fraud cases. This may indicate class imbalance or bad model choice.")

    out_file = "predictions.csv"
    df.to_csv(out_file, index=False)
    print(f"\n[SAVED] Predictions written to {out_file}")


if __name__ == "__main__":
    main()
