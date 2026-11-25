"""
Compare models and choose the threshold on validation data.

Run from the project folder:
    python -m src.compare

Models: logistic regression, linear SVM (calibrated), Complement Naive Bayes.
The threshold is chosen on a validation split of the TRAIN set (best F1
for the negative class) and then applied once to the test set.
"""

import os

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

from src.split import RANDOM_STATE, get_split


def best_f1_threshold(y, scores):
    """Threshold with the highest F1 on the given data."""
    grid = np.arange(0.1, 0.91, 0.02)
    f1s = [f1_score(y, scores >= t) for t in grid]
    return grid[int(np.argmax(f1s))]


def main():
    train, test = get_split()
    os.makedirs("outputs", exist_ok=True)
    y_train = (train["sentiment"] == 0).astype(int).to_numpy()
    y_test = (test["sentiment"] == 0).astype(int).to_numpy()

    # Validation split inside train, again by user
    inner = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    fit_idx, val_idx = next(inner.split(train, groups=train["UserId"]))

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=5,
                                 max_features=300_000, sublinear_tf=True)
    X_fit = vectorizer.fit_transform(train["text"].iloc[fit_idx])
    X_val = vectorizer.transform(train["text"].iloc[val_idx])
    y_fit, y_val = y_train[fit_idx], y_train[val_idx]

    def make_models():
        return {
            "Logistic regression": LogisticRegression(
                max_iter=1000, C=4, random_state=RANDOM_STATE),
            "Linear SVM": CalibratedClassifierCV(
                LinearSVC(C=0.5, random_state=RANDOM_STATE), cv=3),
            "Complement NB": ComplementNB(alpha=0.3),
        }

    # Step 1: threshold per model on validation data
    thresholds = {}
    for name, model in make_models().items():
        model.fit(X_fit, y_fit)
        thresholds[name] = best_f1_threshold(y_val, model.predict_proba(X_val)[:, 1])

    # Step 2: refit on the full train set, evaluate once on test
    X_train = vectorizer.fit_transform(train["text"])
    X_test = vectorizer.transform(test["text"])

    print(f"{'Model':20s} | thr  | precision | recall |   F1  | PR-AUC")
    best_name, best_pr, best_pack = None, -1, None
    for name, model in make_models().items():
        model.fit(X_train, y_train)
        scores = model.predict_proba(X_test)[:, 1]
        flag = scores >= thresholds[name]
        pr = average_precision_score(y_test, scores)
        print(f"{name:20s} | {thresholds[name]:.2f} | "
              f"{precision_score(y_test, flag):9.3f} | "
              f"{recall_score(y_test, flag):6.3f} | "
              f"{f1_score(y_test, flag):.3f} | {pr:.3f}")
        if pr > best_pr:
            best_name, best_pr = name, pr
            best_pack = {"vectorizer": vectorizer, "model": model,
                         "threshold": float(thresholds[name]), "name": name}

    joblib.dump(best_pack, "outputs/best_sentiment.joblib")
    print(f"\nBest by PR-AUC: {best_name} ({best_pr:.3f}), saved to outputs/")


if __name__ == "__main__":
    main()