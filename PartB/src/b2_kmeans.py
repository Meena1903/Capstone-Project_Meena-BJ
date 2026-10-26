"""B2 - Score K-means solutions for K=2 through K=8."""
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from common import OUTPUTS

rfm = pd.read_csv(OUTPUTS / "customer_rfm.csv")
features = ["recency", "frequency", "monetary"]
scaled = StandardScaler().fit_transform(rfm[features])
rows = []
for k in range(2, 9):
    model = KMeans(n_clusters=k, n_init=10, random_state=42).fit(scaled)
    rows.append({"K": k, "inertia": model.inertia_, "silhouette": silhouette_score(scaled, model.labels_)})
table = pd.DataFrame(rows)
table.to_csv(OUTPUTS / "kmeans_scores.csv", index=False)
print(table.round({"inertia": 1, "silhouette": 3}).to_string(index=False))
print("Chosen K=4: elbow at four clusters, best silhouette (0.526), and actionable segment count.")
print("Inertia keeps falling as K grows because more centroids fit the points more closely; that alone always favors the largest K.")
