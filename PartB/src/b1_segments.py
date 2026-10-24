"""B1-B3 - RFM table, K-means, segment profile/naming, PCA, customer_segments.csv.
Run from repo root:  python b1_segments.py"""
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
FINAL_K = 4
os.makedirs("outputs", exist_ok=True)
pd.set_option("display.width", 140)

orders = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
done = orders[orders["status"] == "completed"]

# ---------------- B1. RFM table ----------------
g = done.groupby("customer_id")
rfm = pd.DataFrame({
    "last_order_ts": g["order_ts"].max(),
    "frequency": g["order_id"].nunique(),
    "monetary": g["order_amount"].sum(),
})
rfm["recency"] = (SNAPSHOT - rfm["last_order_ts"]).dt.days
rfm = rfm[["recency", "frequency", "monetary"]].reset_index()

print("=== B1. RFM table ===")
print("customers:", len(rfm), "(ref 1175)")
print("recency  median/min/max:", rfm.recency.median(), rfm.recency.min(), rfm.recency.max(), "(ref 19 / 0 / 180)")
print("frequency mean/max   :", round(rfm.frequency.mean(), 2), rfm.frequency.max(), "(ref 9.46 / 33)")
print("monetary  mean/max   :", round(rfm.monetary.mean(), 2), round(rfm.monetary.max(), 2), "(ref 2552.06 / 14766.55)")

# ---------------- B2. K-means for K = 2..8 ----------------
X = StandardScaler().fit_transform(rfm[["recency", "frequency", "monetary"]])
rows = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    rows.append({"K": k, "inertia": round(km.inertia_, 1), "silhouette": round(silhouette_score(X, km.labels_), 3)})
table = pd.DataFrame(rows)
print("\n=== B2. K-means table ===")
print(table.to_string(index=False))
print("reference: K2 1820.8/0.465 | K3 1268.2/0.445 | K4 743.1/0.526 | K5 617.4/0.481 | K6 503.2/0.482 | K7 408.5/0.416 | K8 366.5/0.412")
print(f"\nChosen K = {FINAL_K}")

# ---------------- B3. profile, name, project ----------------
km = KMeans(n_clusters=FINAL_K, n_init=10, random_state=42).fit(X)
rfm["segment_id"] = km.labels_
prof = rfm.groupby("segment_id").agg(size=("customer_id", "count"),
                                     mean_recency=("recency", "mean"),
                                     mean_frequency=("frequency", "mean"),
                                     mean_monetary=("monetary", "mean")).round(2)

# Cluster numbers are only labels (no order), so names are assigned from the profile, never from the id.
remaining = set(prof.index)
names, actions = {}, {}
premium_id = prof["mean_monetary"].idxmax()
names[premium_id], actions[premium_id] = "Premium Big Spenders", "VIP perks, early access to Electronics, dedicated support"
remaining.discard(premium_id)
loyal_id = prof.loc[list(remaining), "mean_frequency"].idxmax()
names[loyal_id], actions[loyal_id] = "Loyal Regulars", "Loyalty rewards and subscriptions to protect habit"
remaining.discard(loyal_id)
risk_id = prof.loc[list(remaining), "mean_recency"].idxmax()
names[risk_id], actions[risk_id] = "At-Risk / Fading", "Win-back offers before they are lost"
remaining.discard(risk_id)
(low_id,) = remaining
names[low_id], actions[low_id] = "Occasional Low-Value", "Low-cost nudges, bundles to raise basket size"

prof["segment_name"] = prof.index.map(names)
prof["recommended_action"] = prof.index.map(actions)
prof["is_premium"] = prof.index == premium_id
print("\n=== B3a/b. Segment profile (cluster ids are labels only, no ordering) ===")
print(prof.to_string())
print("reference K=4 sizes ~436/138/419/182; 138-seg monetary ~7130, freq ~6.8; 419-seg freq ~19.4, recency ~8.8")

pca = PCA(n_components=2).fit(X)
evr = pca.explained_variance_ratio_
print("\n=== B3c. PCA ===")
print("explained_variance_ratio_:", evr.round(3), "total:", round(evr.sum(), 3), "(ref 0.634 / 0.218 / 0.852)")

out = rfm.copy()
out["segment_name"] = out["segment_id"].map(names)
out["is_premium"] = out["segment_id"] == premium_id
out["monetary"] = out["monetary"].round(2)
out.to_csv("outputs/customer_segments.csv", index=False)
print("\nsaved outputs/customer_segments.csv:", out.shape, "| premium customers:", int(out["is_premium"].sum()))
