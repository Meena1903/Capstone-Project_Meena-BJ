"""B3 - Profile, name, project and save the K=4 customer segments."""
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from common import OUTPUTS

rfm = pd.read_csv(OUTPUTS / "customer_rfm.csv")
columns = ["recency", "frequency", "monetary"]
scaled = StandardScaler().fit_transform(rfm[columns])
rfm["segment_id"] = KMeans(n_clusters=4, n_init=10, random_state=42).fit_predict(scaled)
profile = rfm.groupby("segment_id").agg(
    size=("customer_id", "count"), mean_recency=("recency", "mean"),
    mean_frequency=("frequency", "mean"), mean_monetary=("monetary", "mean"),
).round(2)
premium = profile.mean_monetary.idxmax()
remaining = set(profile.index) - {premium}
loyal = profile.loc[list(remaining), "mean_frequency"].idxmax()
remaining.remove(loyal)
risk = profile.loc[list(remaining), "mean_recency"].idxmax()
remaining.remove(risk)
(low,) = remaining
names = {premium: "Premium Big Spenders", loyal: "Loyal Regulars",
         risk: "At-Risk / Fading", low: "Occasional Low-Value"}
actions = {premium: "VIP perks, early access to Electronics, dedicated support",
           loyal: "Loyalty rewards and subscriptions to protect habit",
           risk: "Win-back offers before they are lost",
           low: "Low-cost nudges and bundles to raise basket size"}
profile["segment_name"] = profile.index.map(names)
profile["recommended_action"] = profile.index.map(actions)
profile["is_premium"] = profile.index == premium
print("Cluster numbers are labels only; they have no ordering.")
print(profile.to_string())
pca = PCA(n_components=2).fit(scaled)
print("PCA explained_variance_ratio_:", pca.explained_variance_ratio_.round(3),
      "total:", round(pca.explained_variance_ratio_.sum(), 3))
rfm["segment_name"] = rfm.segment_id.map(names)
rfm["is_premium"] = rfm.segment_id == premium
rfm["monetary"] = rfm.monetary.round(2)
rfm.to_csv(OUTPUTS / "customer_segments.csv", index=False)
print("saved outputs/customer_segments.csv", rfm.shape)
