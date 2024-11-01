import pandas as pd
df = pd.read_csv("data/test_fraud_dataset.csv", encoding="utf-8-sig")
print("Columns read:", df.columns.tolist())
print(df.head(3))