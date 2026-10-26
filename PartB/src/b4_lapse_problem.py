"""B4 - Build the pre-cutoff lapse cohort, labels and feature table."""
import pandas as pd

from common import CUTOFF, DATA, OUTPUTS, WINDOW_END

orders = pd.read_csv(DATA / "orders.csv", parse_dates=["order_ts"])
customers = pd.read_csv(DATA / "customers.csv", parse_dates=["signup_date"])
products = pd.read_csv(DATA / "products.csv")
orders = orders.merge(products[["product_id", "category"]], on="product_id", how="left")
before = orders[orders.order_ts < CUTOFF]
completed = before[before.status == "completed"]
group = completed.groupby("customer_id")
features = pd.DataFrame({
    "recency": (CUTOFF - group.order_ts.max()).dt.days,
    "frequency": group.order_id.nunique(),
    "monetary": group.order_amount.sum(),
    "avg_discount": group.discount_pct.mean(),
    "avg_delivery": group.delivery_minutes.mean(),
    "fav_category": group.category.agg(lambda values: values.value_counts().sort_index().idxmax()),
})
final = before[before.status.isin(["completed", "cancelled"])]
features["cancel_rate"] = (final.status == "cancelled").groupby(final.customer_id).mean().reindex(features.index).fillna(0)
features = features.join(customers.set_index("customer_id")[["city", "age", "gender", "signup_date"]])
features["tenure_days"] = (CUTOFF - features.signup_date).dt.days
features = features.drop(columns="signup_date")
sept = orders[(orders.status == "completed") & (orders.order_ts >= CUTOFF) & (orders.order_ts < WINDOW_END)]
features["lapsed"] = (~features.index.isin(sept.customer_id)).astype(int)
features.index.name = "customer_id"
features.to_csv(OUTPUTS / "lapse_features.csv")
print("B4 cohort:", len(features), "lapsed:", int(features.lapsed.sum()),
      f"({features.lapsed.mean() * 100:.2f}%)")
print("Features use only orders before 2025-09-01; September orders supply labels only.")
print("saved outputs/lapse_features.csv")
