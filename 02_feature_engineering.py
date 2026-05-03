import pandas as pd
import os

RAW_PATH = "data/KDS Transactions.xlsx"
OUT_PATH = "data/df_customers.csv"

CUTOFF_DATE = pd.Timestamp("2026-03-31")
ACTIVE_WINDOW_DAYS_MONTHLY = 35
ACTIVE_WINDOW_DAYS_ANNUAL = 400

PRICE_ERAS = [
    ("era_99",  pd.Period("2023-11", "M"), pd.Period("2024-08", "M"), 99.0),
    ("era_169", pd.Period("2024-09", "M"), pd.Period("2024-10", "M"), 169.0),
    ("era_199", pd.Period("2024-11", "M"), pd.Period("2025-08", "M"), 199.0),
    ("era_249", pd.Period("2025-09", "M"), pd.Period("2099-12", "M"), 249.0),
]

def assign_era(entry_month):
    for label, start, end, _ in PRICE_ERAS:
        if start <= entry_month <= end:
            return label
    return "unknown"

def era_official_price(era_label):
    for label, _, _, price in PRICE_ERAS:
        if label == era_label:
            return price
    return None

print("Loading raw data...")
df = pd.read_excel(RAW_PATH)
df.columns = ["date", "client", "amount"]
df = df.sort_values(["client", "date"]).reset_index(drop=True)

print("Building df_customers...")

agg = df.groupby("client").agg(
    entry_date=("date", "min"),
    last_payment=("date", "max"),
    n_transactions=("amount", "count"),
    total_revenue=("amount", "sum"),
    entry_price=("amount", "first"),
).reset_index()
agg.rename(columns={"client": "client_id"}, inplace=True)

agg["entry_month"] = agg["entry_date"].dt.to_period("M")
agg["entry_era"] = agg["entry_month"].apply(assign_era)
agg["official_price"] = agg["entry_era"].apply(era_official_price)

agg["is_annual"] = agg["entry_price"] > 700.0
price_deviation = (
    abs(agg["entry_price"] - agg["official_price"]) / agg["official_price"]
)
agg["is_promo"] = ~agg["is_annual"] & (price_deviation > 0.05)

agg["tenure_days"] = (agg["last_payment"] - agg["entry_date"]).dt.days
agg["tenure_months"] = agg["tenure_days"] / 30.44

active_window = agg["is_annual"].map(
    {True: ACTIVE_WINDOW_DAYS_ANNUAL, False: ACTIVE_WINDOW_DAYS_MONTHLY}
)
agg["is_active"] = (
    agg["last_payment"] >= CUTOFF_DATE - pd.to_timedelta(active_window, unit="D")
)
agg["churn_event"] = (~agg["is_active"]).astype(int)

price_consistency = (
    df.groupby("client")["amount"].nunique().rename("unique_amounts").reset_index()
)
price_consistency = price_consistency.rename(columns={"client": "client_id"})
agg = agg.merge(price_consistency, on="client_id")
agg["price_changed"] = agg["unique_amounts"] > 1

agg["km_duration"] = agg["tenure_months"].clip(lower=0.01)
agg["km_event"] = agg["churn_event"]

cols_ordered = [
    "client_id", "entry_date", "entry_month", "entry_price", "official_price",
    "entry_era", "is_promo", "is_annual", "price_changed", "unique_amounts",
    "n_transactions", "last_payment", "total_revenue",
    "tenure_days", "tenure_months", "is_active", "churn_event",
    "km_duration", "km_event",
]
agg = agg[cols_ordered]

os.makedirs("data", exist_ok=True)
agg.to_csv(OUT_PATH, index=False)

print(f"\ndf_customers saved to {OUT_PATH}")
print(f"Shape: {agg.shape}")
print(f"\nColumn summary:")
print(agg.dtypes.to_string())

print("\n--- SEGMENT COUNTS ---")
print(f"  Total customers:    {len(agg)}")
print(f"  Monthly plans:      {(~agg['is_annual']).sum()}")
print(f"  Annual plans:       {agg['is_annual'].sum()}")
print(f"  Promo customers:    {agg['is_promo'].sum()}")
print(f"  Price-changers:     {agg['price_changed'].sum()}")
print(f"  Currently active:   {agg['is_active'].sum()}")
print(f"  Churned:            {agg['churn_event'].sum()}")

print("\n--- ERA BREAKDOWN ---")
era_summary = agg.groupby("entry_era").agg(
    customers=("client_id", "count"),
    monthly=("is_annual", lambda x: (~x).sum()),
    annual=("is_annual", "sum"),
    promo=("is_promo", "sum"),
    active=("is_active", "sum"),
    median_ltv=("total_revenue", "median"),
).reset_index()
print(era_summary.to_string(index=False))
