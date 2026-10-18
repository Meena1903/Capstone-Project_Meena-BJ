"""A5 - Run the 7 queries in sql/queries.sql against data/instamart.db and check the reference results.
Run from repo root:  python a4_sql.py"""
import re, sqlite3
import pandas as pd

text = open("sql/queries.sql").read()
parts = re.split(r"^-- (Q\d):.*$", text, flags=re.M)        # ['', 'Q1', sql, 'Q2', sql, ...]
queries = {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}

con = sqlite3.connect("data/instamart.db")
res = {q: pd.read_sql_query(sql, con) for q, sql in queries.items()}
con.close()
pd.set_option("display.width", 140)

print("Q1 customers:", len(res["Q1"]), "(ref 30)")
q2 = res["Q2"]
print("Q2 rows:", len(q2), "| total:", round(q2.revenue.sum(), 2), "| top:", q2.iloc[0].tolist(), "(ref 30 / 2998668.80 / Bengaluru Electronics 233909.45)")
print("Q3 products:", len(res["Q3"]), "(ref 11)")
print("Q4 customers:", len(res["Q4"]), "(ref 25)")
q5 = res["Q5"]
print("Q5 customers:", len(q5), "| avg:", q5.avg_spend.iloc[0], "(ref 550 / 2552.06)")
q6 = res["Q6"]
print("Q6 top per city:\n", q6[q6.row_num == 1][["city", "customer_id", "total_spend"]].to_string(index=False))
print("Q6 rows with RANK != ROW_NUMBER:", int((q6.rnk != q6.row_num).sum()), "(ref 8)")
q7 = res["Q7"]
print("Q7 days:", len(q7), "| last cumulative:", q7.cumulative_revenue.iloc[-1], "(ref 183 / 2998668.80)")
