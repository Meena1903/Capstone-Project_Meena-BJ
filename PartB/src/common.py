"""Shared paths and preparation helpers for the independently runnable B sections."""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "clean"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)
SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
CUTOFF = pd.Timestamp("2025-09-01")
WINDOW_END = pd.Timestamp("2025-10-01")
NUMERIC = ["recency", "frequency", "monetary", "avg_discount", "cancel_rate",
           "avg_delivery", "age", "tenure_days"]
CATEGORICAL = ["fav_category", "city", "gender"]


def make_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL),
    ])


def load_split():
    import joblib
    return joblib.load(OUTPUTS / "lapse_split.joblib")
