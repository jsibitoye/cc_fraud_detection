import pandas as pd
import numpy as np

np.random.seed(42)

N = 250000  # total rows
fraud_ratio = 0.20  # 20% fraud (more than before)

fraud_n = int(N * fraud_ratio)
nonfraud_n = N - fraud_n

merchants = ["ShopX", "ShopY", "LuxuryStore", "OnlineMall", "ElectroMart", "PharmaCare"]
categories = ["Groceries", "Electronics", "Luxury", "Healthcare", "Travel", "Gaming"]
card_types = ["Visa", "Mastercard", "VirtualCard"]
locations = ["Lagos", "Abuja", "Kano", "PortHarcourt", "Ibadan", "Enugu"]

def generate_transactions(n, fraud=False):
    data = []
    for _ in range(n):
        if fraud:
            amount = np.random.choice(
                [np.random.randint(300000, 1000000),  # high value
                 np.random.randint(10000, 300000)],   # mid value
                p=[0.8, 0.2]
            )
            category = np.random.choice(categories, p=[0.05, 0.35, 0.25, 0.05, 0.15, 0.15])
            merchant = np.random.choice(merchants, p=[0.05, 0.05, 0.25, 0.25, 0.25, 0.15])
            card_type = np.random.choice(card_types, p=[0.3, 0.5, 0.2])
            location = np.random.choice(locations)
            time = np.random.choice([f"{h:02d}:{m:02d}" for h in range(0, 24) for m in [0,30]])
            is_night = 1 if int(time.split(":")[0]) in list(range(22,24)) + list(range(0,6)) else 0
            high_amount_flag = 1 if amount > 300000 else 0
            merchant_risk = 1 if merchant in ["LuxuryStore", "OnlineMall", "ElectroMart", "Gaming"] else 0
            card_risk = 1 if card_type in ["Mastercard", "VirtualCard"] else 0
            fraudflag = 1
        else:
            amount = np.random.choice(
                [np.random.randint(500, 50000), np.random.randint(50000, 200000)],
                p=[0.8, 0.2]
            )
            category = np.random.choice(categories)
            merchant = np.random.choice(merchants)
            card_type = np.random.choice(card_types, p=[0.6, 0.3, 0.1])
            location = np.random.choice(locations)
            time = np.random.choice([f"{h:02d}:{m:02d}" for h in range(0, 24) for m in [0,30]])
            is_night = 1 if int(time.split(":")[0]) in list(range(22,24)) + list(range(0,6)) else 0
            high_amount_flag = 1 if amount > 300000 else 0
            merchant_risk = 1 if merchant in ["LuxuryStore", "OnlineMall"] else 0
            card_risk = 1 if card_type == "VirtualCard" else 0
            fraudflag = 0

        data.append({
            "Amount": amount,
            "Merchant": merchant,
            "Category": category,
            "CardType": card_type,
            "Location": location,
            "Time": time,
            "FraudFlag": fraudflag,
            "HighAmountFlag": high_amount_flag,
            "IsNightTransaction": is_night,
            "MerchantRisk": merchant_risk,
            "CardRisk": card_risk
        })
    return data

fraud_data = generate_transactions(fraud_n, fraud=True)
nonfraud_data = generate_transactions(nonfraud_n, fraud=False)

df = pd.DataFrame(fraud_data + nonfraud_data)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

df.to_csv("data/nigeria_credit_card_fraud_dataset_v4.csv", index=False)
print(f"✅ Generated dataset: {df.shape[0]} rows, Fraud cases = {df['FraudFlag'].sum()}")
