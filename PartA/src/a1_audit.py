"""A1 - Audit the raw data. Run: python PartA/src/a1_audit.py"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"

# ---- 1. Standard load: shape, dtypes, missing values per column ----------
tables = {
    "orders":    pd.read_csv(RAW / "orders.csv"),
    "customers": pd.read_csv(RAW / "customers.csv"),
    "products":  pd.read_csv(RAW / "products.csv"),
}
for name, df in tables.items():
    print(f"\n=== {name}: shape {df.shape} ===")
    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "missing_%": (df.isna().mean() * 100).round(2),
    })
    print(summary)

# ---- 2. Raw-string load: keeps "N/A" and "" exactly as written ------------
# (pandas' default read turns the text "N/A" into NaN, which would hide the
#  120 unparseable amounts, so for the specific audit figures we read as text)
o = pd.read_csv(RAW / "orders.csv", dtype=str, keep_default_na=False)
c = pd.read_csv(RAW / "customers.csv", dtype=str, keep_default_na=False)

amount_num = pd.to_numeric(o["order_amount"], errors="coerce")
deliv_num = pd.to_numeric(o["delivery_minutes"], errors="coerce")
age_num = pd.to_numeric(c["age"], errors="coerce")

audit = {
    "raw orders rows": len(o),
    "exact duplicate rows": int(o.duplicated().sum()),
    "rows with status pending": int((o["status"] == "pending").sum()),
    "distinct order_id": int(o["order_id"].nunique()),
    "non-numeric order_amount": int((amount_num.isna() & (o["order_amount"] != "")).sum()),
    "negative delivery_minutes": int((deliv_num < 0).sum()),
    "blank delivery_minutes": int((o["delivery_minutes"] == "").sum()),
    "customers: blank age": int((c["age"] == "").sum()),
    "customers: age outside 10-100": int(((age_num < 10) | (age_num > 100)).sum()),
    "customers: distinct raw city spellings": int(c["city"].nunique()),
    "customers: distinct non-blank gender spellings": int(c.loc[c["gender"] != "", "gender"].nunique()),
    "customers: blank gender": int((c["gender"] == "").sum()),
}
reference = [13084, 119, 656, 12309, 120, 106, 2541, 61, 6, 24, 10, 30]

print("\n=== AUDIT FIGURES ===")
result = pd.DataFrame({"figure": audit.keys(), "found": audit.values(), "reference": reference})
result["match"] = result["found"] == result["reference"]
print(result.to_string(index=False))
print("\nALL AUDIT FIGURES MATCH" if result["match"].all() else "\nMISMATCH - check the rows above")
