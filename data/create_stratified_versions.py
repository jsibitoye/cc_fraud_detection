import argparse
from pathlib import Path
import pandas as pd


def stratified_sample(df, target_col, n, seed=42):
    fraud = df[df[target_col] == 1]
    legit = df[df[target_col] == 0]

    fraud_ratio = fraud.shape[0] / df.shape[0]

    fraud_n = round(n * fraud_ratio)
    legit_n = n - fraud_n

    if fraud_n > len(fraud):
        raise ValueError(
            f"Requested {fraud_n} fraud rows, but only {len(fraud)} available."
        )
    if legit_n > len(legit):
        raise ValueError(
            f"Requested {legit_n} legit rows, but only {len(legit)} available."
        )

    fraud_sample = fraud.sample(n=fraud_n, random_state=seed, replace=False)
    legit_sample = legit.sample(n=legit_n, random_state=seed, replace=False)

    combined = (
        pd.concat([fraud_sample, legit_sample])
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )

    return combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", required=True)
    parser.add_argument("--outdir", default="data")
    args = parser.parse_args()

    df = pd.read_csv(args.infile)

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    sizes = {
        "v1": 100000,
        "v2": 200000,
        "v3": 300000,
        "v4": 400000,
        "v5": len(df),  # full master dataset
    }

    summary = []

    for v, size in sizes.items():
        if size == len(df):
            sample = df.copy().reset_index(drop=True)
        else:
            sample = stratified_sample(df, "FraudFlag", size)

        fraud_ratio = sample["FraudFlag"].mean()

        sample.to_csv(outdir / f"dataset_{v}.csv", index=False)

        summary.append({
            "version": v,
            "rows": len(sample),
            "fraud_ratio": round(fraud_ratio, 6)
        })

        print(f"{v}: rows={len(sample)}, fraud_ratio={fraud_ratio:.6f}")

    pd.DataFrame(summary).to_csv(outdir / "dataset_summary.csv", index=False)
    print("[OK] Summary saved -> dataset_summary.csv")


if __name__ == "__main__":
    main()