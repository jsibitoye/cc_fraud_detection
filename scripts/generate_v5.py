# scripts/generate_v5.py
import argparse
import pandas as pd
import numpy as np
from sklearn.utils import resample

parser = argparse.ArgumentParser()
parser.add_argument("--in", dest="infile", required=True)
parser.add_argument("--out", dest="outfile", required=True)
parser.add_argument("--n", type=int, default=350000)
parser.add_argument("--fraud_ratio", type=float, default=0.10)  # e.g. 0.10 for 10%
args = parser.parse_args()

df = pd.read_csv(args.infile)
df.columns = df.columns.str.strip()
if 'FraudFlag' not in df.columns:
    raise SystemExit("input must have FraudFlag column")

n = args.n
desired_fraud = int(n * args.fraud_ratio)
desired_legit = n - desired_fraud

fraud = df[df['FraudFlag'] == 1]
legit = df[df['FraudFlag'] == 0]

# If original fraud > desired_fraud, downsample; else upsample with small noise for numeric cols
if len(fraud) >= desired_fraud:
    fraud_new = resample(fraud, n_samples=desired_fraud, replace=False, random_state=42)
else:
    # upsample fraud by sampling with replacement and add small noise to numeric columns
    fraud_up = resample(fraud, n_samples=desired_fraud, replace=True, random_state=42)
    # tiny jitter for numeric columns
    num_cols = fraud_up.select_dtypes(include=[np.number]).columns.tolist()
    for c in num_cols:
        fraud_up[c] = fraud_up[c] * (1 + np.random.normal(0, 0.001, size=len(fraud_up)))
    fraud_new = fraud_up

# For legit: upsample/downsample to desired_legit
if len(legit) >= desired_legit:
    legit_new = resample(legit, n_samples=desired_legit, replace=False, random_state=42)
else:
    legit_new = resample(legit, n_samples=desired_legit, replace=True, random_state=42)

out_df = pd.concat([fraud_new, legit_new]).sample(frac=1, random_state=42).reset_index(drop=True)
out_df.to_csv(args.outfile, index=False)
print("Saved", args.outfile, "rows:", len(out_df), "fraud ratio:", out_df['FraudFlag'].mean())
