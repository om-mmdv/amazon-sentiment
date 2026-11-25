"""
Train the final model for the dashboard and save it to models/.

Run from the project folder:
    python -m src.export_model

Logistic regression is used because every word has a weight, so the
dashboard can explain which words drove a verdict. The threshold is
chosen on a validation part of the train set (best F1 for negatives).
"""

import os

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit

from src.compare import best_f1_threshold
from src.split import RANDOM_STATE, get_split


def main():
    train, _ = get_split()
    y = (train["sentiment"] == 0).astype(int).to_numpy()

    def make_vectorizer():
        # Smaller vocabulary keeps the file small enough for GitHub
        return TfidfVectorizer(ngram_range=(1, 2), min_df=5,
                               max_features=150_000, sublinear_tf=True)

    def make_model():
        return LogisticRegression(max_iter=1000, C=4, random_state=RANDOM_STATE)

    # Threshold on a validation split (by user), never on the test set
    inner = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    fit_idx, val_idx = next(inner.split(train, groups=train["UserId"]))
    vec = make_vectorizer()
    model = make_model().fit(vec.fit_transform(train["text"].iloc[fit_idx]), y[fit_idx])
    scores = model.predict_proba(vec.transform(train["text"].iloc[val_idx]))[:, 1]
    threshold = float(best_f1_threshold(y[val_idx], scores))

    # Final model on the full train set
    vec = make_vectorizer()
    model = make_model().fit(vec.fit_transform(train["text"]), y)

    os.makedirs("models", exist_ok=True)
    joblib.dump({"vectorizer": vec, "model": model, "threshold": threshold},
                "models/sentiment.joblib", compress=3)
    size = os.path.getsize("models/sentiment.joblib") / 1e6
    print(f"Threshold: {threshold:.2f}. Saved models/sentiment.joblib ({size:.1f} MB)")


if __name__ == "__main__":
    main()