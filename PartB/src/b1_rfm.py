"""B1 - Build the completed-order RFM table and report its checkpoints."""
import pandas as pd

from common import DATA, OUTPUTS, SNAPSHOT

orders = pd.read_csv(DATA / "orders.csv", parse_dates=["order_ts"])
done = orders[orders.status == "completed"]
group = done.groupby("customer_id")
rfm = pd.DataFrame({
    "last_order_ts": group.order_ts.max(),
    "frequency": group.order_id.nunique(),
    "monetary": group.order_amount.sum(),
})
rfm["recency"] = (SNAPSHOT - rfm.last_order_ts).dt.days
rfm = rfm[["recency", "frequency", "monetary"]].reset_index()
rfm.to_csv(OUTPUTS / "customer_rfm.csv", index=False)
print("B1 RFM table; snapshot 2025-10-01 00:00:00")
print("customers:", len(rfm))
print("recency median/min/max:", rfm.recency.median(), rfm.recency.min(), rfm.recency.max())
print("frequency mean/max:", round(rfm.frequency.mean(), 2), rfm.frequency.max())
print("monetary mean/max:", round(rfm.monetary.mean(), 2), round(rfm.monetary.max(), 2))
print("saved outputs/customer_rfm.csv")
