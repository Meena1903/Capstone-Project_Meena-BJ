"""A2-A4 - Clean customers + orders, then store CSVs and SQLite DB.
Run: python PartA/src/a2_clean.py"""
import os, sqlite3
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW, CLEAN, DB = ROOT / "data/raw", ROOT / "data/clean", ROOT / "data/instamart.db"
SUBMISSION_CLEAN = ROOT / "PartA/clean"
CLEAN.mkdir(parents=True, exist_ok=True)
SUBMISSION_CLEAN.mkdir(parents=True, exist_ok=True)
log = {}   # rows/values changed per rule -> copy into README

products = pd.read_csv(RAW / "products.csv")

# ======================= CUSTOMERS =======================
c = pd.read_csv(RAW / "customers.csv", dtype=str, keep_default_na=False)

# (a) city: strip + title case
before = c["city"].copy()
c["city"] = c["city"].str.strip().str.title()
log["cust: city values changed"] = int((before != c["city"]).sum())

# (b) gender -> male / female / unknown
g = c["gender"].str.strip().str.lower()
c["gender"] = g.map(lambda x: "male" if x in ("m", "male") else "female" if x in ("f", "female") else "unknown")

# (c) age: outside 10-100 -> missing, then fill with city median (groupby + transform)
c["age"] = pd.to_numeric(c["age"], errors="coerce")
bad_age = (c["age"] < 10) | (c["age"] > 100)
log["cust: ages set to missing (outside 10-100)"] = int(bad_age.sum())
c.loc[bad_age, "age"] = float("nan")
log["cust: ages filled with city median"] = int(c["age"].isna().sum())
c["age"] = c["age"].fillna(c.groupby("city")["age"].transform("median"))

# (d) signup_date -> datetime
c["signup_date"] = pd.to_datetime(c["signup_date"])

# ======================= ORDERS =======================
o = pd.read_csv(RAW / "orders.csv", dtype=str, keep_default_na=False)
for col in ["quantity", "discount_pct"]:
    o[col] = pd.to_numeric(o[col])
log["orders: raw rows"] = len(o)

# 1. order_ts -> datetime
o["order_ts"] = pd.to_datetime(o["order_ts"])

# 2. remove exact duplicate rows
n = len(o); o = o.drop_duplicates()
log["orders: exact duplicates removed"] = n - len(o)

# 3. per order_id keep the record with the latest order_ts
n = len(o)
o = o.sort_values(["order_id", "order_ts"]).drop_duplicates("order_id", keep="last")
log["orders: older records of same order_id removed"] = n - len(o)

# 4. order_amount -> numeric; fill missing from price * qty * (1 - disc/100)
o["order_amount"] = pd.to_numeric(o["order_amount"], errors="coerce")
o = o.merge(products[["product_id", "category", "price_inr"]], on="product_id", how="left")
miss = o["order_amount"].isna()
log["orders: amounts filled from price"] = int(miss.sum())
o.loc[miss, "order_amount"] = (o["price_inr"] * o["quantity"] * (1 - o["discount_pct"] / 100)).round(2)[miss]

# 5. delivery_minutes: negative -> missing; completed -> category median; cancelled stay missing
o["delivery_minutes"] = pd.to_numeric(o["delivery_minutes"], errors="coerce")   # "" -> NaN
neg = o["delivery_minutes"] < 0
log["orders: negative delivery set to missing"] = int(neg.sum())
o.loc[neg, "delivery_minutes"] = float("nan")
comp = o["status"] == "completed"
med = o[comp].groupby("category")["delivery_minutes"].transform("median")   # median among completed only
fill = comp & o["delivery_minutes"].isna()
log["orders: completed delivery filled with category median"] = int(fill.sum())
o.loc[fill, "delivery_minutes"] = med[fill]

# keep the same 9 columns as the raw table (category / price live in products)
o = o.drop(columns=["price_inr", "category"]).sort_values("order_ts").reset_index(drop=True)

# ======================= STORE =======================
c.to_csv(CLEAN / "customers.csv", index=False)
products.to_csv(CLEAN / "products.csv", index=False)
o.to_csv(CLEAN / "orders.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
# Keep the explicit Part A submission paths in sync with the canonical files
# consumed by Parts B and D.
c.to_csv(SUBMISSION_CLEAN / "customers.csv", index=False)
products.to_csv(SUBMISSION_CLEAN / "products.csv", index=False)
o.to_csv(SUBMISSION_CLEAN / "orders.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")

o_db, c_db = o.copy(), c.copy()
o_db["order_ts"] = o_db["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")   # text in DB
c_db["signup_date"] = c_db["signup_date"].dt.strftime("%Y-%m-%d")
if DB.exists():
    DB.unlink()
with sqlite3.connect(DB) as con:
    c_db.to_sql("customers", con, index=False)
    products.to_sql("products", con, index=False)
    o_db.to_sql("orders", con, index=False)

print("=== CLEANING LOG (copy into README) ===")
for k, v in log.items():
    print(f"{k:60s} {v}")
print("\nshapes:", "customers", c.shape, "| orders", o.shape, "| products", products.shape)
print("status counts:", o["status"].value_counts().to_dict())
print("saved:", CLEAN / "*.csv", "and", DB)
