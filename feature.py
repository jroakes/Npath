"""Feature engineering functions."""

from typing import List, Tuple
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from prefixspan import PrefixSpan


def check_sequence(sequence, sub_sequence):
    """Check if sub_sequence is present in sequence."""
    it = iter(sequence)
    return all(c in it for c in sub_sequence)


def find_frequent_sequences(
    sequences: List[List[str]], min_freq: int = 3, min_len: int = 3
) -> List[Tuple[int, List[str]]]:
    """Find frequent sequences using PrefixSpan.

    Parameters
    ----------
    sequences : List[List[str]]
        List of sequences.
    min_freq : int, optional
        Minimum frequency for a sequence to be considered frequent, by default 3.
    min_len : int, optional
        Minimum length for a sequence to be considered, by default 3.

    Returns
    -------
    List[Tuple[int, List[str]]]
        List of tuples where each tuple contains the frequency and the sequence.
    """
    ps = PrefixSpan(sequences)
    ps.minlen = min_len
    return ps.frequent(min_freq, closed=True)


def plot_important_features_prefixspan(
    df: pd.DataFrame, top_n: int = 10, min_freq: int = 10, min_seq_len: int = 3
) -> None:
    """Plot important features using PrefixSpan.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data.
    top_n : int, optional
        The number of important features to plot, by default 10.
    min_freq : int, optional
        The minimum frequency of the sequence, by default 10.
    min_seq_len : int, optional
        The minimum length of the sequence, by default 3.

    Returns
    -------
    None
    """

    # Ensure 'activity_list' and 'converted' columns exist in DataFrame
    if "activity_list" not in df.columns or "converted" not in df.columns:
        raise ValueError(
            "DataFrame must contain 'activity_list' and 'converted' columns."
        )

    # Discover frequent sequences using PrefixSpan
    sequences = df["activity_list"].tolist()

    frequent_sequences = find_frequent_sequences(
        sequences, min_freq=min_freq, min_len=min_seq_len
    )

    # Prepare an empty dictionary to hold data
    new_data = {}

    # Create binary features for each frequent sequence
    for idx, item in enumerate(frequent_sequences):
        sub_seq = item[1]  # Assuming item[1] contains the sequence
        new_data[f"seq_{idx}"] = df["activity_list"].apply(
            lambda x: int(check_sequence(x, sub_seq))
        )

    # Concatenate all new sequence columns to the original DataFrame
    df = pd.concat([df, pd.DataFrame(new_data)], axis=1)

    # Selecting the sequence columns
    seq_cols = [col for col in df.columns if col.startswith("seq_")]

    # Prepare data for logistic regression
    X = df[seq_cols]
    y = df["converted"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    # Perform logistic regression
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    feature_importance = log_reg.coef_[0]

    # Create DataFrame for feature importance
    features_df = pd.DataFrame(
        {
            "Sequence": [item[1] for item in frequent_sequences],
            "Importance": feature_importance,
        }
    )

    # Sort features based on importance
    features_df = features_df.sort_values(by="Importance", ascending=False)

    # Plot the top N important features using a horizontal bar chart
    top_features_df = features_df.head(top_n).copy()
    top_features_df["Sequence"] = top_features_df["Sequence"].apply(
        lambda x: " -> ".join(x)
    )
    top_features_df.plot(x="Sequence", y="Importance", kind="barh", figsize=(10, 10))
    plt.title(f"Top {top_n} Important Sequences")
    plt.xlabel("Importance")
    plt.ylabel("Sequence")
    plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature at the top
    plt.show()


# Path: feature.py
