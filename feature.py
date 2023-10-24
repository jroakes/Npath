import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from mlxtend.preprocessing import TransactionEncoder
from prefixspan import PrefixSpan
from mlxtend.frequent_patterns import apriori, association_rules
from sequential.seq2pat import Seq2Pat, Attribute


def check_sequence(sequence, sub_sequence):
    it = iter(sequence)
    return all(c in it for c in sub_sequence)


def plot_important_features_apriori(df: pd.DataFrame, top_n: int = 10):
    # Convert sequences to binary matrix using mlxtend
    sequences = df["activity_list"].tolist()
    te = TransactionEncoder()
    te_ary = te.fit(sequences).transform(sequences)
    df_bin = pd.DataFrame(te_ary, columns=te.columns_)

    # Discover frequent itemsets using apriori
    frequent_itemsets = apriori(
        df_bin, min_support=0.07, use_colnames=True
    )  # Adjust min_support as necessary

    # Create binary features for each frequent itemset
    for idx, item in enumerate(frequent_itemsets["itemsets"]):
        sub_seq = list(item)
        df[f"seq_{idx}"] = df["activity_list"].apply(
            lambda x: check_sequence(x, sub_seq)
        )

    # Selecting the sequence columns
    seq_cols = [col for col in df.columns if col.startswith("seq_")]

    # Prepare data for logistic regression
    X = df[seq_cols]
    y = df["converted"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Perform logistic regression
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    feature_importance = log_reg.coef_[0]

    # Create DataFrame for feature importance
    features_df = pd.DataFrame(
        {
            "Sequence": [
                list(frequent_itemsets["itemsets"][i])
                for i in range(len(frequent_itemsets["itemsets"]))
            ],
            "Importance": feature_importance,
        }
    )

    # Sort features based on importance
    features_df = features_df.sort_values(by="Importance", ascending=False)

    # Plot the top N important features
    top_features_df = features_df.head(top_n).copy()
    top_features_df["Sequence"] = top_features_df["Sequence"].apply(
        lambda x: " -> ".join(x)
    )
    ax = top_features_df.plot(
        x="Sequence", y="Importance", kind="barh", figsize=(10, 6)
    )
    plt.title(f"Top {top_n} Important Sequences")
    plt.ylabel("Importance")
    plt.show()


def plot_important_features_seq2pat(df: pd.DataFrame, top_n: int = 10):
    # Discover frequent sequences using Seq2Pat
    sequences = df["activity_list"].tolist()
    seq2pat = Seq2Pat(sequences=sequences)
    patterns = seq2pat.get_patterns(
        min_frequency=10
    )  # Adjust the threshold as necessary

    # Prepare an empty dictionary to hold data
    new_data = {}

    # Create binary features for each frequent sequence
    for idx, item in enumerate(patterns):
        sub_seq = item[:-1]  # Exclude the frequency at the end of each item
        new_data[f"seq_{idx}"] = df["activity_list"].apply(
            lambda x: check_sequence(x, sub_seq)
        )

    # Concatenate all new sequence columns to the original DataFrame
    df = pd.concat([df, pd.DataFrame(new_data)], axis=1)

    # Selecting the sequence columns
    seq_cols = [col for col in df.columns if col.startswith("seq_")]

    # Prepare data for logistic regression
    X = df[seq_cols]
    y = df["converted"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Perform logistic regression
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    feature_importance = log_reg.coef_[0]

    # Create DataFrame for feature importance
    features_df = pd.DataFrame(
        {
            "Sequence": [patterns[i][:-1] for i in range(len(patterns))],
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
    top_features_df.plot(x="Sequence", y="Importance", kind="barh", figsize=(10, 6))
    plt.title(f"Top {top_n} Important Sequences")
    plt.xlabel("Importance")  # Note that the x and y labels have switched places
    plt.show()


def plot_important_features_prefixspan(df: pd.DataFrame, top_n: int = 10):
    """Plot important features using PrefixSpan."""

    # Discover frequent sequences using PrefixSpan
    sequences = df["activity_list"].tolist()

    ps = PrefixSpan(sequences)
    ps.minlen = 2  # Adjust the minimum length as necessary
    frequent_sequences = ps.frequent(
        20, closed=True
    )  # Adjust the threshold as necessary

    # Prepare an empty dictionary to hold data
    new_data = {}

    # Create binary features for each frequent sequence
    for idx, item in enumerate(frequent_sequences):
        sub_seq = item[1]  # Assuming item[1] contains the sequence
        new_data[f"seq_{idx}"] = df["activity_list"].apply(
            lambda x: check_sequence(x, sub_seq)
        )

    # Concatenate all new sequence columns to the original DataFrame
    df = pd.concat([df, pd.DataFrame(new_data)], axis=1)

    # Selecting the sequence columns
    seq_cols = [col for col in df.columns if col.startswith("seq_")]

    # Prepare data for logistic regression
    X = df[seq_cols]
    y = df["converted"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Perform logistic regression
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    feature_importance = log_reg.coef_[0]

    # Create DataFrame for feature importance
    features_df = pd.DataFrame(
        {
            "Sequence": [
                item[1] for item in frequent_sequences
            ],  # Changed patterns to frequent_sequences
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
    top_features_df.plot(x="Sequence", y="Importance", kind="barh", figsize=(10, 6))
    plt.title(f"Top {top_n} Important Sequences")
    plt.xlabel("Importance")  # Note that the x and y labels have switched places
    plt.show()
