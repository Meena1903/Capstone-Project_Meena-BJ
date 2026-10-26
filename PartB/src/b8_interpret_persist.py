"""B8 - Interpret the tuned models, persist the selected model and save a retention list."""
import joblib
import numpy as np
import pandas as pd

from common import OUTPUTS, load_split

data = load_split()
models = joblib.load(OUTPUTS / "lapse_tuned_models.joblib")
X_test, y_test = data["X_test"], data["y_test"]
lr_search, rf_search = models["Logistic Regression"]["search"], models["Random Forest"]["search"]
rf = rf_search.best_estimator_
feature_names = rf.named_steps["prep"].get_feature_names_out()
importance = pd.Series(rf.named_steps["clf"].feature_importances_, index=feature_names).sort_values(ascending=False)
print("Top five Random Forest features:\n", importance.head(5).round(4).to_string())
lr = lr_search.best_estimator_
coefficients = pd.Series(lr.named_steps["clf"].coef_[0], index=lr.named_steps["prep"].get_feature_names_out())
print("Logistic Regression coefficients (positive means greater lapse probability):")
for feature in ("num__recency", "num__frequency", "num__cancel_rate"):
    print(feature, round(coefficients[feature], 3))
selected = rf if rf_search.best_score_ >= lr_search.best_score_ else lr
joblib.dump(selected, OUTPUTS / "lapse_model.joblib")
reloaded = joblib.load(OUTPUTS / "lapse_model.joblib")
assert np.array_equal(selected.predict(X_test), reloaded.predict(X_test))
print("Reloaded test predictions reproduce exactly.")
segments = pd.read_csv(OUTPUTS / "customer_segments.csv")[["customer_id", "is_premium"]]
retention = pd.DataFrame({"customer_id": X_test.index,
                          "lapse_probability": selected.predict_proba(X_test)[:, 1].round(4),
                          "actually_lapsed": y_test.to_numpy()})
retention = retention.merge(segments, on="customer_id")
retention = retention[retention.is_premium].sort_values("lapse_probability", ascending=False).head(10)
retention[["customer_id", "lapse_probability", "actually_lapsed"]].to_csv(OUTPUTS / "retention_list.csv", index=False)
print("Premium test customers:", int(segments.set_index("customer_id").loc[X_test.index, "is_premium"].sum()),
      "| retention rows:", len(retention), "| truly lapsed:", int(retention.actually_lapsed.sum()))
print("Retention list is illustrative, not a leakage-free evaluation: the Premium flag uses whole-window RFM.")
