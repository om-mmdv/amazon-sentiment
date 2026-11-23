"""
Exploratory analysis of the Amazon review data.

Run from the project folder:
    python -m src.explore

Checks class balance, duplicated texts (a leak risk), review length and
how concentrated reviews are per user and per product. Saves plots to assets/.
"""

import os

import matplotlib.pyplot as plt

from src.data import load_reviews


def print_summary(df):
    """Print duplicates and concentration statistics."""
    print("=== Score distribution (1-2 negative, 4-5 positive) ===")
    print(df["Score"].value_counts().sort_index())

    dup_text = df["Text"].duplicated().sum()
    print(f"\n=== Duplicated review texts ===")
    print(f"  {dup_text} of {len(df)} ({dup_text / len(df) * 100:.1f}%)")

    users = df["UserId"].value_counts()
    products = df["ProductId"].value_counts()
    print("\n=== Concentration ===")
    print(f"  Unique users: {len(users)}, unique products: {len(products)}")
    print(f"  Reviews by the most active user: {users.iloc[0]}")
    print(f"  Reviews of the most reviewed product: {products.iloc[0]}")

    words = df["Text"].str.split().str.len()
    print("\n=== Review length (words) by sentiment ===")
    print(words.groupby(df["sentiment"]).describe().round(1))


def plot_length(df):
    """Histogram of review length in words, per class (capped at 300)."""
    words = df["Text"].str.split().str.len()
    plt.figure(figsize=(8, 5))
    for label, name, color in [(0, "Negative", "#f97316"), (1, "Positive", "#3b82f6")]:
        plt.hist(words[df["sentiment"] == label], bins=60, range=(0, 300),
                 density=True, alpha=0.6, label=name, color=color)
    plt.xlabel("Words per review (capped at 300 on the plot)")
    plt.ylabel("Share of reviews")
    plt.title("Review length by sentiment")
    plt.legend()
    plt.tight_layout()
    plt.savefig("assets/review_length.png", dpi=150)
    plt.close()


def plot_scores(df):
    """Bar chart of the number of reviews per star rating."""
    counts = df["Score"].value_counts().sort_index()
    colors = ["#f97316" if s <= 2 else "#3b82f6" for s in counts.index]
    plt.figure(figsize=(6, 4))
    plt.bar(counts.index.astype(str), counts.values, color=colors)
    plt.xlabel("Stars")
    plt.ylabel("Reviews")
    plt.title("Reviews per star rating (3 stars removed)")
    plt.tight_layout()
    plt.savefig("assets/score_distribution.png", dpi=150)
    plt.close()


def main():
    os.makedirs("assets", exist_ok=True)
    df = load_reviews()

    print_summary(df)

    print("\n=== Three random negative reviews ===")
    for text in df[df["sentiment"] == 0]["Text"].sample(3, random_state=1):
        print(f"  - {text[:250]}...")

    plot_length(df)
    plot_scores(df)
    print("\nPlots saved to assets/")


if __name__ == "__main__":
    main()