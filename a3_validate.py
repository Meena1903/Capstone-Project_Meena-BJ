"""A4 - Validate the clean data in data/instamart.db against the brief's checkpoints.
Run from repo root:  python a3_validate.py"""
import sqlite3
import pandas as pd

con = sqlite3.connect("data/instamart.db")
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
