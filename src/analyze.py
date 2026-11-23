"""
Error and word analysis of the TF-IDF + logistic regression model.

Run from the project folder:
    python -m src.analyze

Shows which words push a review toward "negative" or "positive" and
prints examples of the model's most confident mistakes.
"""

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np

from src.split import get_split


def plot_top_words(words, weights, title, filename, color):
    """Horizontal bar chart of the strongest words."""
    plt.figure(figsize=(7, 6))
    plt.barh(words[::-1], np.abs(weights)[::-1], color=color)
    plt.xlabel("Weight in the model")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"assets/{filename}", dpi=150)
    plt.close()


def main():
    os.makedirs("assets", exist_ok=True)
    _, test = get_split()
    model = joblib.load("outputs/baseline_model.joblib")

    vectorizer = model.steps[0][1]
    classifier = model.steps[1][1]
    names = np.array(vectorizer.get_feature_names_out())
    weights = classifier.coef_[0]  # positive weight = pushes toward NEGATIVE

    order = np.argsort(weights)
    neg_idx, pos_idx = order[::-1][:15], order[:15]

    print("=== Words pushing toward NEGATIVE ===")
    print(", ".join(names[neg_idx]))
    print("\n=== Words pushing toward POSITIVE ===")
    print(", ".join(names[pos_idx]))

    plot_top_words(names[neg_idx], weights[neg_idx],
                   "Top words for negative reviews", "top_negative.png", "#f97316")
    plot_top_words(names[pos_idx], weights[pos_idx],
                   "Top words for positive reviews", "top_positive.png", "#3b82f6")

    # Mistakes: most confident wrong predictions
    y = (test["sentiment"] == 0).astype(int).to_numpy()
    scores = model.predict_proba(test["text"])[:, 1]
    texts = test["text"].to_numpy()

    false_neg = np.where((y == 1) & (scores < 0.5))[0]
    false_pos = np.where((y == 0) & (scores >= 0.5))[0]
    print(f"\nMissed negatives: {len(false_neg)}, false alarms: {len(false_pos)}")

    print("\n=== Negative reviews the model missed (lowest negative score) ===")
    for i in false_neg[np.argsort(scores[false_neg])[:3]]:
        print(f"  [{scores[i]:.2f}] {texts[i][:220]}...")

    print("\n=== Positive reviews wrongly flagged (highest negative score) ===")
    for i in false_pos[np.argsort(-scores[false_pos])[:3]]:
        print(f"  [{scores[i]:.2f}] {texts[i][:220]}...")

    print("\nPlots saved to assets/")


if __name__ == "__main__":
    main()
