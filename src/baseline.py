"""
Baseline: TF-IDF + logistic regression.

Run from the project folder:
    python -m src.baseline

TF-IDF turns each review into numbers: words that are frequent in the review
but rare overall get a high weight. Logistic regression then learns which
words push the sentiment up or down.
"""

import os

import joblib
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import make_pipeline

from src.split import RANDOM_STATE, get_split


def report(name, y_true, p_negative, y_pred_negative):
    """Metrics for the NEGATIVE class (the one we want to find)."""
    print(f"{name:22s} | precision {precision_score(y_true, y_pred_negative, zero_division=0):.3f} "
          f"| recall {recall_score(y_true, y_pred_negative):.3f} "
          f"| F1 {f1_score(y_true, y_pred_negative):.3f} "
          f"| PR-AUC {average_precision_score(y_true, p_negative):.3f}")


def main():
    train, test = get_split()
    os.makedirs("outputs", exist_ok=True)

    # We treat "negative review" as the target class (1 = negative)
    y_train = (train["sentiment"] == 0).astype(int)
    y_test = (test["sentiment"] == 0).astype(int)

    # Baseline 1: always "not negative"
    dummy = DummyClassifier(strategy="most_frequent").fit(train["text"], y_train)
    report("Always positive", y_test, dummy.predict_proba(test["text"])[:, 1],
           dummy.predict(test["text"]))

    # Baseline 2: TF-IDF (single words and word pairs) + logistic regression
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=300_000,
                        sublinear_tf=True),
        LogisticRegression(max_iter=1000, C=4, class_weight="balanced",
                           random_state=RANDOM_STATE),
    )
    model.fit(train["text"], y_train)
    scores = model.predict_proba(test["text"])[:, 1]
    report("TF-IDF + LogReg", y_test, scores, (scores >= 0.5).astype(int))

    joblib.dump(model, "outputs/baseline_model.joblib")
    print("\nSaved to outputs/baseline_model.joblib")


if __name__ == "__main__":
    main()