"""B7 - Oversample lapsed training rows and compare on the unchanged test set."""
import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from common import OUTPUTS, load_split

data = load_split()
models = joblib.load(OUTPUTS / "lapse_tuned_models.joblib")
X_train, X_test, y_train, y_test = (data[k] for k in ("X_train", "X_test", "y_train", "y_test"))
lr = models["Logistic Regression"]["search"].best_estimator_
n0, n1 = int((y_train == 0).sum()), int((y_train == 1).sum())
extra = X_train[y_train == 1].sample(n=n0 - n1, replace=True, random_state=42)
X_balanced = pd.concat([X_train, extra])
y_balanced = pd.concat([y_train, pd.Series(1, index=extra.index)])
balanced = clone(lr).fit(X_balanced, y_balanced)
print(f"Training rows balanced only: {n0}/{n1} -> {(y_balanced == 0).sum()}/{(y_balanced == 1).sum()}")
for label, model in (("Original", lr), ("Oversampled", balanced)):
    predicted = model.predict(X_test)
    print(label, "confusion [[TN FP], [FN TP]]:\n", confusion_matrix(y_test, predicted))
    print("precision/recall/F1:", *(round(m(y_test, predicted, zero_division=0), 3)
                                    for m in (precision_score, recall_score, f1_score)))
print("Oversampling is likely to raise recall and lower precision: more lapsing customers are found, with more false alarms.")
