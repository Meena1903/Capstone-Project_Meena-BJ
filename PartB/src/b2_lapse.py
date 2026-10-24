"""B4-B8 - Lapse prediction: cohort/features, leakage-free preprocessing, tuning, evaluation,
class-balance experiment, interpretation, persistence, retention list.
Run AFTER b1_segments.py, from repo root:  python b2_lapse.py"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CUTOFF, WINDOW_END = pd.Timestamp("2025-09-01"), pd.Timestamp("2025-10-01")

# Resolve paths relative to this script's location so the data/ and outputs/
# directories are found regardless of the current working directory.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(_REPO_ROOT, "data", "clean")
OUTPUTS_DIR = os.path.join(_REPO_ROOT, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)
pd.set_option("display.width", 140)

orders = pd.read_csv(os.path.join(DATA_DIR, "orders.csv"), parse_dates=["order_ts"])
customers = pd.read_csv(os.path.join(DATA_DIR, "customers.csv"), parse_dates=["signup_date"])
products = pd.read_csv(os.path.join(DATA_DIR, "products.csv"))
orders = orders.merge(products[["product_id", "category"]], on="product_id", how="left")

# =============== B4. Cohort, label, features ===============
before = orders[orders["order_ts"] < CUTOFF]                       # ONLY orders before the cutoff feed features
before_done = before[before["status"] == "completed"]
sept_done = orders[(orders["status"] == "completed") & (orders["order_ts"] >= CUTOFF) & (orders["order_ts"] < WINDOW_END)]

g = before_done.groupby("customer_id")
feat = pd.DataFrame({
    "recency": (CUTOFF - g["order_ts"].max()).dt.days,
    "frequency": g["order_id"].nunique(),
    "monetary": g["order_amount"].sum(),
    "avg_discount": g["discount_pct"].mean(),
    "avg_delivery": g["delivery_minutes"].mean(),
    "fav_category": g["category"].agg(lambda s: s.value_counts().sort_index().idxmax()),
})
final = before[before["status"].isin(["completed", "cancelled"])]
feat["cancel_rate"] = (final["status"] == "cancelled").groupby(final["customer_id"]).mean()
feat["cancel_rate"] = feat["cancel_rate"].fillna(0)
feat = feat.join(customers.set_index("customer_id")[["city", "age", "gender", "signup_date"]])
feat["tenure_days"] = (CUTOFF - feat["signup_date"]).dt.days
feat = feat.drop(columns="signup_date")
feat["lapsed"] = (~feat.index.isin(sept_done["customer_id"])).astype(int)

NUM = ["recency", "frequency", "monetary", "avg_discount", "cancel_rate", "avg_delivery", "age", "tenure_days"]
CAT = ["fav_category", "city", "gender"]
X, y = feat[NUM + CAT], feat["lapsed"]
print("=== B4. Cohort ===")
print("cohort:", len(feat), "(ref 1156) | lapsed:", int(y.sum()), f"({y.mean()*100:.2f}%) (ref 443 / 38.32%)")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print("train/test rows:", len(X_train), len(X_test), "(ref 924 / 232)")
base_pred = np.zeros(len(y_test), dtype=int)
print("baseline (always not lapsed): accuracy", round(accuracy_score(y_test, base_pred), 3),
      "| lapsed F1", f1_score(y_test, base_pred, zero_division=0), "(ref 0.616 / 0)")

# =============== B5. Preprocessing without leakage ===============
def make_prep():
    return ColumnTransformer([("num", StandardScaler(), NUM),
                              ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT)])

prep_check = make_prep().fit(X_train)                                # fitted on TRAIN rows only
learned = prep_check.named_transformers_["num"].mean_[NUM.index("frequency")]
print("\n=== B5. Leakage check ===")
print(f"scaler mean of frequency (train only): {learned:.3f} | full-cohort mean: {feat['frequency'].mean():.3f}")
print("They differ because the scaler saw only the 924 training rows, not the whole cohort (test rows were never used).")
assert X.isna().sum().sum() == 0
print("Missing values in feature columns:", int(X.isna().sum().sum()), "(must be 0)")

# =============== B6. Tune and evaluate ===============
lr_pipe = Pipeline([("prep", make_prep()), ("clf", LogisticRegression(max_iter=1000))])
lr_grid = GridSearchCV(lr_pipe, {"clf__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1").fit(X_train, y_train)
rf_pipe = Pipeline([("prep", make_prep()), ("clf", RandomForestClassifier(n_estimators=100, random_state=42))])
rf_grid = GridSearchCV(rf_pipe, {"clf__max_depth": [3, 5, 8], "clf__min_samples_leaf": [5, 10, 20]},
                       cv=5, scoring="f1", n_jobs=-1).fit(X_train, y_train)
print("\n=== B6. Grid searches (cv=5, scoring=f1) ===")
print("LogReg best:", lr_grid.best_params_, "CV F1", round(lr_grid.best_score_, 4))
print("RandForest best:", rf_grid.best_params_, "CV F1", round(rf_grid.best_score_, 4), "(45 fits x 5 folds)")

def report(name, model, Xt, yt):
    pred = model.predict(Xt)
    m = {"model": name,
         "precision": round(precision_score(yt, pred, zero_division=0), 3),
         "recall": round(recall_score(yt, pred, zero_division=0), 3),
         "f1": round(f1_score(yt, pred, zero_division=0), 3)}
    print(f"\n{name}: confusion matrix [[TN FP],[FN TP]]\n{confusion_matrix(yt, pred)}")
    print({k: v for k, v in m.items() if k != 'model'})
    return m

print("\n=== Test-set evaluation (lapsed = positive class) ===")
lr_m = report("Logistic Regression", lr_grid.best_estimator_, X_test, y_test)
rf_m = report("Random Forest", rf_grid.best_estimator_, X_test, y_test)
# Select on CROSS-VALIDATED F1 (training data only) so the test set never influences the choice.
best_name, best_model = (("Random Forest", rf_grid.best_estimator_) if rf_grid.best_score_ >= lr_grid.best_score_
                         else ("Logistic Regression", lr_grid.best_estimator_))
sel = rf_m if best_name == "Random Forest" else lr_m
print(f"\nSelected model: {best_name} (chosen by CV F1: RF {rf_grid.best_score_:.3f} vs LR {lr_grid.best_score_:.3f}); "
      f"its test F1 = {sel['f1']} (required >= 0.55)")
print("(ref: LogReg C=1 F1~0.62; RandForest depth 8 / leaf 10 F1~0.65)")

# =============== B7. Class-balance experiment (train only) ===============
n0, n1 = (y_train == 0).sum(), (y_train == 1).sum()
extra = X_train[y_train == 1].sample(n=n0 - n1, replace=True, random_state=42)
X_bal = pd.concat([X_train, extra]); y_bal = pd.concat([y_train, pd.Series(1, index=extra.index)])
print(f"\n=== B7. Over-sampling (train only): {n0}/{n1} -> {(y_bal==0).sum()}/{(y_bal==1).sum()} ===")
lr_bal = clone(lr_grid.best_estimator_).fit(X_bal, y_bal)
print("--- un-balanced LogReg ---"); a = report("LogReg (original)", lr_grid.best_estimator_, X_test, y_test)
print("--- balanced LogReg ---");    b = report("LogReg (balanced)", lr_bal, X_test, y_test)
moves = {k: round(b[k] - a[k], 3) for k in ("precision", "recall", "f1")}
print("change after balancing:", moves, "(ref: recall ~0.64 -> ~0.85, precision stays ~0.6)")

# =============== B8. Interpret, persist, use ===============
names = rf_grid.best_estimator_.named_steps["prep"].get_feature_names_out()
imp = pd.Series(rf_grid.best_estimator_.named_steps["clf"].feature_importances_, index=names).sort_values(ascending=False)
print("\n=== B8a. Top-5 Random Forest features ===")
print(imp.head(5).round(4).to_string())
coef = pd.Series(lr_grid.best_estimator_.named_steps["clf"].coef_[0],
                 index=lr_grid.best_estimator_.named_steps["prep"].get_feature_names_out())
print("\nLogReg coefficients (positive = higher lapse probability):")
for f in ["num__recency", "num__frequency", "num__cancel_rate"]:
    print(f"  {f:20s} {coef[f]:+.3f}  -> higher value makes lapse {'MORE' if coef[f] > 0 else 'LESS'} likely")

joblib.dump(best_model, os.path.join(OUTPUTS_DIR, "lapse_model.joblib"))
reloaded = joblib.load(os.path.join(OUTPUTS_DIR, "lapse_model.joblib"))
same = np.array_equal(best_model.predict(X_test), reloaded.predict(X_test)) and \
       np.allclose(best_model.predict_proba(X_test), reloaded.predict_proba(X_test))
print("\n=== B8b. Persistence ===\nreloaded model reproduces test predictions exactly:", same)
assert same

seg = pd.read_csv(os.path.join(OUTPUTS_DIR, "customer_segments.csv"))[["customer_id", "is_premium"]]
res = pd.DataFrame({"customer_id": X_test.index,
                    "lapse_probability": best_model.predict_proba(X_test)[:, 1].round(4),
                    "actually_lapsed": y_test.values})
res = res.merge(seg, on="customer_id", how="inner")
prem = res[res["is_premium"]].sort_values("lapse_probability", ascending=False).head(10)
prem[["customer_id", "lapse_probability", "actually_lapsed"]].to_csv(os.path.join(OUTPUTS_DIR, "retention_list.csv"), index=False)
print("\n=== B8c. Retention list (Premium customers in the test set) ===")
print(prem[["customer_id", "lapse_probability", "actually_lapsed"]].to_string(index=False))
print(f"Premium customers in test set: {int(res['is_premium'].sum())} | rows written: {len(prem)} | truly lapsed: {int(prem['actually_lapsed'].sum())}")
print("NOTE: the Premium flag uses whole-window RFM (includes September), so this list is illustrative, not a leakage-free evaluation.")
