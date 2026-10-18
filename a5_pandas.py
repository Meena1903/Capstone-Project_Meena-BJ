"""A6 - pandas cross-check of Q2 + exploratory analysis of completed order_amount.
Run from repo root:  python a5_pandas.py"""
import sqlite3
import pandas as pd

con = sqlite3.connect("data/instamart.db")
df = pd.read_sql_query("""SELECT o.*, c.city, p.category
                          FROM orders o
                          JOIN customers c ON o.customer_id = c.customer_id
                          JOIN products  p ON o.product_id  = p.product_id""", con)
con.close()
done = df[df["status"] == "completed"]
pd.set_option("display.width", 140)

# ---- (a) pivot_table with margins = Q2 rebuilt in pandas ------------------
pv = pd.pivot_table(done, values="order_amount", index="city", columns="category",
                    aggfunc="sum", margins=True, margins_name="Total").round(2)
print("=== PIVOT (city x category, INR) ===\n", pv)
print("\nPivot grand total:", pv.loc["Total", "Total"], "| SQL total: 2998668.80 ->",
      "EQUAL" if round(pv.loc["Total", "Total"], 2) == 2998668.80 else "DIFFERENT")

city = done.groupby("city").agg(total=("order_amount", "sum"),
                                avg=("order_amount", "mean"),
                                orders=("order_amount", "count")).round(2)
print("\n=== PER-CITY sum / mean / count ===\n", city)

# ---- (b) distribution of completed order_amount ---------------------------
x = done["order_amount"]
q1, q3 = x.quantile(0.25), x.quantile(0.75)
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
stats = {"mean": x.mean(), "median": x.median(), "skewness": x.skew(),
         "Q1": q1, "Q3": q3, "lower fence": lo, "upper fence": hi,
         "outliers below": int((x < lo).sum()), "outliers above": int((x > hi).sum())}
print("\n=== order_amount statistics ===")
for k, v in stats.items():
    print(f"{k:15s}{v:12.2f}")

corr = done[["order_amount", "discount_pct", "quantity", "delivery_minutes"]].corr().round(3)
print("\n=== correlation matrix ===\n", corr)
from itertools import combinations
pairs = {(a, b): corr.loc[a, b] for a, b in combinations(corr.columns, 2)}
ranked = sorted(pairs.items(), key=lambda kv: abs(kv[1]), reverse=True)
print("\nlargest |corr| pair:", ranked[0][0], "=", ranked[0][1])
print("runner-up        :", ranked[1][0], "=", ranked[1][1])
