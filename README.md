# Instamart Customer Intelligence & Support Copilot

## Overview
One repository, four connected Parts built on synthetic (fictional) Instamart data. All money is in INR.
- **Part A** audits and cleans the raw exports and stores them in SQLite (`data/instamart.db`).
- **Part B** segments customers (RFM + K-means) and predicts lapse.
- **Part C** triages support tickets with privacy and output guardrails.
- **Part D** answers policy questions (RAG) and runs a refund agent.

All language-model calls use the offline `mock_llm.py`. No API key or network is needed. `generate_data.py` and `mock_llm.py` are unmodified.

## Reproduce
Run from the repository root (Python 3.11+).
1. `python -m venv venv`
2. `source venv/bin/activate`   (Windows: `venv\Scripts\activate`)
3. `pip install pandas numpy scikit-learn joblib`
4. `pip freeze > requirements.txt`
5. `python generate_data.py`
6. `python check_setup.py`   (prints SETUP OK)
7. `python a1_audit.py`   (Part A audit)
8. `python a2_clean.py`   (Part A cleaning, CSVs and database)
9. `python a3_validate.py`   (prints ALL CHECKPOINTS PASSED)
10. `python a4_sql.py`   (runs `sql/queries.sql`)
11. `python a5_pandas.py`   (pandas cross-check and exploratory analysis)

<!-- Part B, C, D commands are added here as they are built -->

## Part A

### A1. Audit of the raw data
Shapes: orders 13,084 x 9; customers 1,200 x 5; products 40 x 4. Missing values in a standard pandas read: orders `order_amount` 53 (0.41%), `delivery_minutes` 2,541 (19.42%); customers `age` 61 (5.08%), `gender` 30 (2.50%); no other column has missing values. Note: pandas reads the text `N/A` as missing, so the 53 are only the `N/A` entries; the other 67 bad amounts are the text `error`. The figures below are therefore taken from a text read (`dtype=str, keep_default_na=False`).

| Audit figure | Found |
|---|---|
| Raw orders rows | 13,084 |
| Exact duplicate rows | 119 |
| Rows with status pending | 656 |
| Distinct order_id | 12,309 |
| Non-numeric order_amount | 120 |
| Negative delivery_minutes | 106 |
| Blank delivery_minutes | 2,541 |
| Customers: blank age | 61 |
| Customers: age outside 10-100 | 6 |
| Customers: distinct raw city spellings | 24 |
| Customers: distinct non-blank gender spellings | 10 |
| Customers: blank gender | 30 |

### A2. Cleaning rules and what each changed
**Customers**
| Rule | Changed |
|---|---|
| City: strip spaces, title case (24 spellings become 6 cities) | 476 values |
| Gender: m/male to male, f/female to female, blank to unknown | 30 unknown |
| Age outside 10-100 set to missing | 6 values |
| Missing age filled with the median age of the customer's city (`groupby` + `transform`) | 67 values (61 blank + 6 invalid) |
| `signup_date` converted to datetime | all 1,200 rows |

Result: Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184; female 610, male 560, unknown 30; 0 missing ages, all between 18 and 60.

**Orders** (applied in this order)
| # | Rule | Changed |
|---|---|---|
| 1 | `order_ts` converted to datetime | all rows |
| 2 | Exact duplicate rows removed | 119 rows |
| 3 | Per `order_id` keep only the record with the latest `order_ts` (drops the earlier pending records) | 656 rows |
| 4 | `order_amount` converted to numeric (unparseable become missing), then filled with `round(price_inr x quantity x (1 - discount_pct/100), 2)` | 112 values |
| 5a | Negative `delivery_minutes` set to missing | 105 values |
| 5b | Completed orders: missing `delivery_minutes` filled with the median of that product category among completed orders (`groupby` + `transform`) | 774 values |
| 5c | Cancelled orders: `delivery_minutes` left missing (never delivered, so a delivery time would be invented data) | 1,199 stay missing |

Counts for rules 4 and 5a are lower than the audit (120 and 106) because rule 2 had already removed duplicate rows that carried some of those bad values.

Result: 12,309 rows (one per order_id); 11,110 completed, 1,199 cancelled, 0 pending; 0 missing `order_amount`; completed revenue INR 2,998,668.80.

### A3. Storage and validation
`a2_clean.py` writes `data/clean/customers.csv`, `products.csv`, `orders.csv` and the SQLite database `data/instamart.db` with tables `customers`, `products`, `orders` (`order_ts` stored as text `YYYY-MM-DD HH:MM:SS`). `a3_validate.py` reads the database and checks 14 reference checkpoints (cities, genders, ages, row counts, status counts, missing values, revenue sum). It prints `ALL CHECKPOINTS PASSED`.

### A4. SQL analysis (`sql/queries.sql`, run by `a4_sql.py`)
| Query | Concept | Result |
|---|---|---|
| Q1 | WHERE then HAVING | 30 customers |
| Q2 | Three-table join, revenue by city and category | 30 rows; total INR 2,998,668.80; largest cell Bengaluru / Electronics INR 233,909.45 |
| Q3 | Scalar subquery | 11 products above the average price |
| Q4 | LEFT JOIN | 25 customers without a completed order |
| Q5 | Two CTEs | 550 customers above the average spend of INR 2,552.06 |
| Q6 | RANK, DENSE_RANK, ROW_NUMBER per city | Top customers: Bengaluru C0082 (12,856.20), Chennai C0930 (10,151.40), Delhi C0291 (13,742.60), Hyderabad C0743 (12,730.05), Mumbai C0693 (14,766.55), Pune C0426 (10,377.10); 8 rows have RANK different from ROW_NUMBER |
| Q7 | Running total | 183 daily rows; last cumulative value INR 2,998,668.80 |

Why ties make RANK differ from ROW_NUMBER: tied customers share the same RANK (1, 1, 3) while ROW_NUMBER always gives each row a unique number (1, 2, 3).

### A5. pandas cross-check and exploratory analysis
The `pd.pivot_table(..., margins=True)` rebuild of Q2 has a grand total of INR 2,998,668.80, equal to the SQL total.

| City | Sum (INR) | Mean (INR) | Count |
|---|---|---|---|
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Chennai | 463,629.15 | 272.72 | 1,700 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Pune | 500,918.05 | 252.48 | 1,984 |

Completed `order_amount`: mean 269.91, median 135.00, skewness 4.61, Q1 64.80, Q3 297.00, IQR fences -283.50 and 645.30, with 0 outliers below and 897 above.

Correlation matrix (completed orders):

| | order_amount | discount_pct | quantity | delivery_minutes |
|---|---|---|---|---|
| order_amount | 1.000 | -0.095 | 0.087 | -0.003 |
| discount_pct | -0.095 | 1.000 | 0.000 | -0.010 |
| quantity | 0.087 | 0.000 | 1.000 | -0.013 |
| delivery_minutes | -0.003 | -0.010 | -0.013 | 1.000 |

**The four questions** 

**What is this data about?** Quick-commerce orders from six Indian cities, with customers, products and order amounts in INR. **Distributions:** completed order_amount is strongly right-skewed (mean 269.91 against median 135.00, skewness 4.61): most orders are small and a few are large. **Outliers:** 897 orders lie above the upper fence of 645.30 and none below the lower fence; 635 of them (71%) are Electronics and 166 are Dairy, so they look like genuine high-value purchases rather than errors. **Relationships:** all correlations among order_amount, discount_pct, quantity and delivery_minutes are tiny. The largest in absolute value is order_amount with discount_pct (-0.095), just ahead of order_amount with quantity (+0.087). A correlation of this size is very weak and does not establish causation.

## Part B
<!-- to be completed -->

## Part C
<!-- to be completed -->

## Part D
<!-- to be completed -->

## Limitations and Next Steps
<!-- to be completed -->