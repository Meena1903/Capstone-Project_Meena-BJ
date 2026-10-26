"""A3 - Store the clean CSVs in SQLite and validate reference checkpoints.
Run after a2_clean.py: python PartA/src/a3_validate.py"""
from pathlib import Path
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data/instamart.db"
CLEAN = ROOT / "data/clean"
customers = pd.read_csv(CLEAN / "customers.csv", parse_dates=["signup_date"])
products = pd.read_csv(CLEAN / "products.csv")
orders = pd.read_csv(CLEAN / "orders.csv", parse_dates=["order_ts"])

# Store order timestamps as ISO text in SQLite, as required by the SQL tasks.
customers_db = customers.copy()
customers_db["signup_date"] = customers_db["signup_date"].dt.strftime("%Y-%m-%d")
orders_db = orders.copy()
orders_db["order_ts"] = orders_db["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
if DB.exists():
    DB.unlink()
with sqlite3.connect(DB) as con:
    customers_db.to_sql("customers", con, index=False)
    products.to_sql("products", con, index=False)
    orders_db.to_sql("orders", con, index=False)

con = sqlite3.connect(DB)
c = pd.read_sql_query("SELECT * FROM customers", con)
o = pd.read_sql_query("SELECT * FROM orders", con)
con.close()
done = o[o["status"] == "completed"]
canc = o[o["status"] == "cancelled"]

checks = {
    "6 cities": c["city"].nunique() == 6,
    "city counts": c["city"].value_counts().to_dict() == {"Pune": 227, "Delhi": 210, "Bengaluru": 197,
                                                         "Chennai": 191, "Hyderabad": 191, "Mumbai": 184},
    "gender counts": c["gender"].value_counts().to_dict() == {"female": 610, "male": 560, "unknown": 30},
    "0 missing ages": int(c["age"].isna().sum()) == 0,
    "ages within 18-60": bool(c["age"].between(18, 60).all()),
    "12309 order rows": len(o) == 12309,
    "one row per order_id": o["order_id"].is_unique,
    "11110 completed": len(done) == 11110,
    "1199 cancelled": len(canc) == 1199,
    "no pending rows": int((o["status"] == "pending").sum()) == 0,
    "0 missing order_amount": int(o["order_amount"].isna().sum()) == 0,
    "0 missing delivery (completed)": int(done["delivery_minutes"].isna().sum()) == 0,
    "1199 missing delivery (cancelled)": int(canc["delivery_minutes"].isna().sum()) == 1199,
    "completed revenue 2,998,668.80": round(float(done["order_amount"].sum()), 2) == 2998668.80,
}
failed = [k for k, ok in checks.items() if not ok]
for k, ok in checks.items():
    print(("PASS  " if ok else "FAIL  ") + k)
assert not failed, f"Failed: {failed}"
print("ALL CHECKPOINTS PASSED")
