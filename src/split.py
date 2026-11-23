"""
Leak-free train/test split for the review data.

1. Duplicated texts are removed (the same review appears under several
   product ids, which would leak between train and test).
2. The split is made by USER: all reviews of one person go either to
   train or to test, so author style cannot leak.
"""

import re

from sklearn.model_selection import GroupShuffleSplit

from src.data import load_reviews

RANDOM_STATE = 42
TEST_SIZE = 0.2


def clean_text(text):
    """Remove HTML tags such as <br /> and extra spaces."""
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def get_split():
    """Returns train_df, test_df (DataFrames with 'text' and 'sentiment')."""
    df = load_reviews()
    df["text"] = (df["Summary"] + ". " + df["Text"]).map(clean_text)

    before = len(df)
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    print(f"Removed {before - len(df)} duplicated reviews, {len(df)} left")

    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE,
                                 random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(df, groups=df["UserId"]))
    return df.iloc[train_idx], df.iloc[test_idx]


if __name__ == "__main__":
    train, test = get_split()
    for name, part in [("Train", train), ("Test", test)]:
        print(f"{name}: {len(part)} reviews, "
              f"negative share {(part['sentiment'] == 0).mean() * 100:.1f}%, "
              f"users {part['UserId'].nunique()}")
    shared = set(train["UserId"]) & set(test["UserId"])
    print(f"Users present in both parts: {len(shared)}")
