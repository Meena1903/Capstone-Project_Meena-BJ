"""B6 - Tune Logistic Regression and Random Forest, then evaluate the held-out test set."""
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from common import OUTPUTS, make_preprocessor, load_split

data = load_split()
X_train, X_test, y_train, y_test = (data[k] for k in ("X_train", "X_test", "y_train", "y_test"))
lr = GridSearchCV(Pipeline([("prep", make_preprocessor()), ("clf", LogisticRegression(max_iter=1000))]),
                  {"clf__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1").fit(X_train, y_train)
rf = GridSearchCV(Pipeline([("prep", make_preprocessor()),
                           ("clf", RandomForestClassifier(n_estimators=100, random_state=42))]),
                  {"clf__max_depth": [3, 5, 8], "clf__min_samples_leaf": [5, 10, 20]},
                  cv=5, scoring="f1", n_jobs=-1).fit(X_train, y_train)
print("Logistic Regression:", lr.best_params_, "CV F1", round(lr.best_score_, 4))
print("Random Forest:", rf.best_params_, "CV F1", round(rf.best_score_, 4))
results = {}
for name, search in (("Logistic Regression", lr), ("Random Forest", rf)):
    prediction = search.predict(X_test)
    metrics = {"precision": precision_score(y_test, prediction, zero_division=0),
               "recall": recall_score(y_test, prediction, zero_division=0),
               "f1": f1_score(y_test, prediction, zero_division=0)}
    results[name] = {"search": search, "metrics": metrics}
    print(f"\n{name} confusion [[TN FP], [FN TP]]:\n{confusion_matrix(y_test, prediction)}")
    print({key: round(value, 3) for key, value in metrics.items()})
print("Baseline (always not lapsed): accuracy 0.616, positive-class F1 0.000")
joblib.dump(results, OUTPUTS / "lapse_tuned_models.joblib")
print("saved outputs/lapse_tuned_models.joblib")
