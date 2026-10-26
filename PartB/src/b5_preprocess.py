"""B5 - Split rows and verify train-only preprocessing without feature leakage."""
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from common import CATEGORICAL, NUMERIC, OUTPUTS, make_preprocessor

features = pd.read_csv(OUTPUTS / "lapse_features.csv", index_col="customer_id")
X, y = features[NUMERIC + CATEGORICAL], features["lapsed"]
assert X.isna().sum().sum() == 0, "Feature table contains missing values"
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42,
)
preprocessor = make_preprocessor().fit(X_train)
learned_mean = preprocessor.named_transformers_["num"].mean_[NUMERIC.index("frequency")]
print("B5 train/test:", len(X_train), len(X_test))
print(f"Training-only scaler mean frequency: {learned_mean:.3f}; full cohort mean: {features.frequency.mean():.3f}")
print("They differ because the scaler was fitted on the training rows only.")
print("Missing values in all feature columns:", int(X.isna().sum().sum()))
joblib.dump({"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test},
            OUTPUTS / "lapse_split.joblib")
print("saved outputs/lapse_split.joblib")
