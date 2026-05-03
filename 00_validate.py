import pandas as pd
import os

RAW_PATH = "data/KDS Transactions.xlsx"
OUT_PATH = "data/validation_report.txt"

lines = []

def log(msg=""):
    print(msg)
    lines.append(msg)

df = pd.read_excel(RAW_PATH)
df.columns = ["date", "client", "amount"]

log("=" * 60)
log("KDS DATA VALIDATION REPORT")
log("=" * 60)

log(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns")

log("\n--- DTYPES ---")
for col, dtype in df.dtypes.items():
    log(f"  {col}: {dtype}")

log("\n--- NULL COUNTS ---")
nulls = df.isnull().sum()
for col, n in nulls.items():
    flag = " *** NULL ISSUE ***" if n > 0 else ""
    log(f"  {col}: {n} nulls{flag}")

log("\n--- DATE RANGE ---")
log(f"  Min date: {df['date'].min()}")
log(f"  Max date: {df['date'].max()}")
expected_min = pd.Timestamp("2023-11-01")
expected_max = pd.Timestamp("2026-04-01")
if df['date'].min() < expected_min:
    log("  *** WARNING: dates earlier than expected ***")
if df['date'].max() > expected_max:
    log("  *** WARNING: dates later than expected ***")

log("\n--- CUSTOMER IDs ---")
log(f"  Unique customers: {df['client'].nunique()}")
log(f"  Min ID: {df['client'].min()}  Max ID: {df['client'].max()}")
if df['client'].min() < 1:
    log("  *** WARNING: non-positive client ID ***")

log("\n--- AMOUNT SANITY ---")
log(f"  Min amount: {df['amount'].min():.2f}")
log(f"  Max amount: {df['amount'].max():.2f}")
log(f"  Negative amounts: {(df['amount'] <= 0).sum()}")
log(f"  Zero amounts: {(df['amount'] == 0).sum()}")
if (df['amount'] <= 0).any():
    log("  *** WARNING: non-positive amounts exist ***")

log("\n--- DUPLICATES ---")
dupes = df.duplicated(subset=["date", "client", "amount"]).sum()
log(f"  Exact duplicate rows (date+client+amount): {dupes}")
same_day = df.duplicated(subset=["date", "client"]).sum()
log(f"  Same client, same timestamp: {same_day}")
if dupes > 0:
    log("  *** WARNING: duplicate rows found ***")

log("\n--- TRANSACTIONS PER CUSTOMER ---")
tx = df.groupby("client").size()
log(f"  Min: {tx.min()}  Max: {tx.max()}  Median: {tx.median()}"
    f"  Mean: {tx.mean():.2f}")
log(
    f"  Customers with only 1 transaction: "
    f"{(tx == 1).sum()} ({(tx == 1).mean() * 100:.1f}%)"
)

log("\n--- AMOUNT DISTRIBUTION ---")
log(f"  Unique amounts: {df['amount'].nunique()}")
log(
    f"  P25: {df['amount'].quantile(0.25):.2f}"
    f"  P50: {df['amount'].quantile(0.50):.2f}"
    f"  P75: {df['amount'].quantile(0.75):.2f}"
)
log(f"  Amounts > 700 (annual plans): {(df['amount'] > 700).sum()} transactions")

log("\n--- MONTHLY NEW CUSTOMERS ---")
first_tx = df.groupby("client")["date"].min().dt.to_period("M")
new_per_month = first_tx.value_counts().sort_index()
for period, count in new_per_month.items():
    log(f"  {period}: {count}")

log("\n--- ZERO-ACQUISITION MONTHS (check) ---")
all_months = pd.period_range(
    start=df["date"].min().to_period("M"),
    end=df["date"].max().to_period("M"),
    freq="M"
)
zero_months = [str(m) for m in all_months if m not in new_per_month.index]
if zero_months:
    log(f"  Months with NO new customers: {zero_months}")
else:
    log("  None — all months have at least one new customer.")

log("\n" + "=" * 60)
log("VALIDATION COMPLETE — review warnings above if any.")
log("=" * 60)

os.makedirs("data", exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"\nReport saved to {OUT_PATH}")
