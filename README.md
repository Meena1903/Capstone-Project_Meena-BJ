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

12. `python b1_segments.py`   (Part B: RFM, K-means, segments, PCA)
13. `python b2_lapse.py`   (Part B: lapse model, balance experiment, joblib, retention list)

<!-- Part C, D commands are added here as they are built -->

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

No encoder, scaler or resampler saw any test row (preprocessing sits inside a scikit-learn `Pipeline`, and over-sampling draws only from training rows). No order dated 2025-09-01 or later entered the lapse features. The segment labels from B3 were not used as features.

### B1. RFM table
Snapshot 2025-10-01 00:00:00, completed orders only. 1,175 customers (those with at least one completed order).

| Measure | Value |
|---|---|
| Recency (days): median / min / max | 19 / 0 / 180 |
| Frequency: mean / max | 9.46 / 33 |
| Monetary (INR): mean / max | 2,552.06 / 14,766.55 |

### B2. K-means (RFM standardised with `StandardScaler`, `n_init=10`, `random_state=42`)
| K | Inertia | Silhouette |
|---|---|---|
| 2 | 1820.8 | 0.465 |
| 3 | 1268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.480 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

**Elbow and choice.** Inertia falls steeply up to K=4 (drops of about 550, 525 and then 125 from K=3 to K=4 to K=5) and flattens afterwards, so the elbow is at K=4. K=4 also has the highest silhouette (0.526), and four segments are few enough for the business to act on, so **K=4** is chosen.

**Why K=8 inertia is lower than K=2 yet cannot pick K.** Adding clusters can only move centres closer to their points, so inertia always shrinks as K grows (it would reach 0 with one cluster per customer). The value alone therefore always favours the largest K; the elbow, the silhouette and business meaning must decide.

### B3. Segments (K=4)
Cluster numbers are only labels with no ordering; names below were assigned from the profile of each cluster, not from its number.

| Segment id | Name | Size | Mean recency (days) | Mean frequency | Mean monetary (INR) | Premium | Recommended action |
|---|---|---|---|---|---|---|---|
| 0 | Occasional Low-Value | 436 | 28.86 | 3.77 | 1,057.62 | No | Low-cost nudges and bundles to raise basket size |
| 1 | Premium Big Spenders | 138 | 24.12 | 6.77 | 7,129.70 | **Yes** | VIP perks, early access to Electronics, dedicated support |
| 2 | Loyal Regulars | 419 | 8.77 | 19.40 | 3,410.02 | No | Loyalty rewards and subscriptions to protect the habit |
| 3 | At-Risk / Fading | 182 | 109.04 | 2.20 | 685.99 | No | Win-back offers before they are lost |

Premium is the segment with the highest mean monetary value (138 customers). PCA on the scaled RFM matrix: explained variance ratio 0.634 and 0.218, total 0.852. `outputs/customer_segments.csv` has columns customer_id, recency, frequency, monetary, segment_id, segment_name, is_premium.

### B4. Lapse problem
Cutoff 2025-09-01. Cohort: every customer with at least one completed order before the cutoff. Label `lapsed` = 1 if the customer has no completed order from 2025-09-01 through 2025-09-30.

Features (built only from orders before the cutoff plus the clean customer table): recency (days from cutoff to last completed order), frequency, monetary, average discount_pct, cancellation rate (cancelled / all orders with a final status), average delivery_minutes of completed orders, favourite product category, city, age, gender, tenure in days (cutoff minus signup_date).

| Item | Value |
|---|---|
| Cohort | 1,156 customers |
| Lapsed | 443 (38.32%) |
| Split | `train_test_split(test_size=0.2, stratify=y, random_state=42)`: 924 train / 232 test |
| Baseline (always "not lapsed") on test | accuracy 0.616, lapsed-class F1 0.0 |

### B5. Preprocessing without leakage
`ColumnTransformer` with `StandardScaler` on the 8 numeric features and `OneHotEncoder(drop="first")` on city, gender and favourite category, inside a `Pipeline` so both are fitted on training rows only. The scaler's learned mean of frequency is 8.115 against a full-cohort mean of 8.029: they differ because the scaler saw only the 924 training rows. No feature column has a missing value (checked in code: 0).

### B6. Tuning and test evaluation
| Model | Grid (cv=5, scoring f1) | Best parameters | Best CV F1 |
|---|---|---|---|
| Logistic Regression (`max_iter=1000`) | C in {0.01, 0.1, 1, 10, 100} | C = 100 | 0.6506 |
| Random Forest (`n_estimators=100`, `random_state=42`) | max_depth in {3, 5, 8} x min_samples_leaf in {5, 10, 20} (45 fits) | max_depth 8, min_samples_leaf 10 | 0.6865 |

Test set, lapsed class (positive):

| Model | Confusion matrix [[TN, FP], [FN, TP]] | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | [[111, 32], [31, 58]] | 0.644 | 0.652 | 0.648 |
| Random Forest | [[111, 32], [32, 57]] | 0.640 | 0.640 | 0.640 |

Both beat the baseline (F1 0.0) and the required 0.55. The reference logistic regression best C was 1; mine is 100 because my feature set differs slightly.

**Model choice: Random Forest.** It was selected on cross-validated F1 (0.687 against 0.651), which keeps the test set out of the decision; on the test set the two models are practically tied (F1 0.640 against 0.648). Missing a customer who is about to lapse (a false negative) loses all of that customer's future revenue, while wrongly targeting a loyal customer (a false positive) costs only the price of a message or a small offer, so recall matters more than precision for the retention team. Our forest still misses 32 of 89 lapsing customers, so in practice the team should lower the decision threshold or use the balanced model below if they want to catch more of them.

### B7. Class-balance experiment
Training set over-sampled with duplicates of lapsed rows (drawn with replacement, `random_state=42`) from 570 / 354 to 570 / 570; the best logistic regression was refit and scored on the unchanged test set.

| Logistic Regression | Precision | Recall | F1 | Confusion matrix |
|---|---|---|---|---|
| Original | 0.644 | 0.652 | 0.648 | [[111, 32], [31, 58]] |
| Balanced | 0.589 | 0.854 | 0.697 | [[90, 53], [13, 76]] |

Recall moved most (+0.202); precision did not rise, it fell (-0.055). For the retention team this means the balanced model finds 76 of 89 lapsing customers instead of 58, at the price of 53 false alarms instead of 32, which is a good trade when the retention offer is cheap.

### B8. Interpretation, persistence and use
Top-5 Random Forest feature importances:

| Feature | Importance | Business reading |
|---|---|---|
| monetary | 0.2778 | How much a customer has spent is the strongest signal of staying power |
| frequency | 0.2768 | Habitual buyers are far less likely to stop |
| recency | 0.1112 | A long gap since the last order is an early warning |
| avg_delivery | 0.0616 | Slower deliveries add some lapse risk |
| avg_discount | 0.0532 | Discount level plays a small role |

Logistic regression coefficients (on standardised features; positive means higher lapse probability):

| Feature | Coefficient | Business reading |
|---|---|---|
| recency | +0.040 | Longer since last order slightly raises lapse risk (weak once frequency is known) |
| frequency | -1.698 | Each extra order lowers lapse risk strongly: the key loyalty driver |
| cancel_rate | -0.124 | Near zero, so no clear effect of cancellations on lapse |

The selected model (Random Forest pipeline) was saved with `joblib.dump` to `outputs/lapse_model.joblib`; the reloaded model reproduces the original test predictions exactly (checked with `np.array_equal`).

`outputs/retention_list.csv`: of the 40 Premium customers in the test set, the 10 with the highest predicted lapse probability were written (columns customer_id, lapse_probability, actually_lapsed). 1 of these 10 truly lapsed. The Premium flag comes from the whole-window RFM of B3, which includes the September window that defines the lapse label, so this retention list is illustrative and not a leakage-free evaluation (the model's own features still use only orders before the cutoff).

## Part C
<!-- to be completed -->

## Part D
<!-- to be completed -->

## Limitations and Next Steps
<!-- to be completed -->