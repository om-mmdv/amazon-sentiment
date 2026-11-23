"""
Data loading for the Amazon review sentiment project.

Every other script gets its data from load_reviews(), so loading and
label creation live in one place.
"""

import os

import pandas as pd

DATA_PATH = os.path.join("data", "Reviews.csv")


def load_reviews(drop_neutral=True):
    """
    Read Reviews.csv and return a clean DataFrame.

    Adds a binary "sentiment" column:
      4-5 stars -> 1 (positive), 1-2 stars -> 0 (negative).
    3-star reviews are neutral and are dropped by default, because they
    do not fit a positive/negative split.
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"File not found: {DATA_PATH}\n"
            "Download Reviews.csv from Kaggle "
            "(search: 'Amazon Fine Food Reviews') and put it into data/."
        )

    df = pd.read_csv(DATA_PATH, usecols=["Id", "ProductId", "UserId",
                                         "Score", "Summary", "Text"])
    df = df.dropna(subset=["Text"])
    df["Summary"] = df["Summary"].fillna("")

    if drop_neutral:
        df = df[df["Score"] != 3]
    df["sentiment"] = (df["Score"] >= 4).astype(int)
    return df.reset_index(drop=True)


if __name__ == "__main__":
    # Quick self-check: run `python -m src.data`
    df = load_reviews()
    print(f"Shape: {df.shape}")
    print(df[["Score", "sentiment", "Summary"]].head())
    print("\nSentiment counts:")
    print(df["sentiment"].value_counts())
    print(f"Negative share: {(df['sentiment'] == 0).mean() * 100:.1f}%")