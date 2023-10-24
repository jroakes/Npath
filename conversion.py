import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.decomposition import PCA
import hdbscan
from sklearn.ensemble import IsolationForest
from prefixspan import PrefixSpan
from itertools import product
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from IPython.display import display


def standardize_data(X):
    """Standardizes the data."""
    scaler = StandardScaler()
    return scaler.fit_transform(X)


def apply_pca(X, n_components=10):
    """Applies PCA to reduce dimensionality."""
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X)


def perform_clustering(X, min_cluster_size=3):
    """Performs clustering using HDBSCAN."""
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    return clusterer.fit_predict(X)


def anomaly_detection(X, contamination=0.05):
    """Performs anomaly detection using Isolation Forest."""
    iso_forest = IsolationForest(contamination=contamination)
    return iso_forest.fit_predict(X)


def print_user_and_activity(df: pd.DataFrame, top_n: int = 5):
    """Prints user and activity."""
    for index, row in df.iterrows():
        print(f"{index+1}. Count: {row['freq']}, Sequence: {row['sequence']}\n")


def convertor_review(df: pd.DataFrame, top_n: int = 10):
    """Review converters and non-converters."""

    # Convert activity_list to binary matrix
    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform(df["activity_list"])

    # Standardize the data
    X_standardized = standardize_data(X)

    # Apply PCA to reduce dimensionality
    Xm = apply_pca(X_standardized)

    # Clustering using HDBSCAN
    df["cluster"] = perform_clustering(Xm)

    # Identify clusters that have converters (-1 is noise)
    converter_clusters = [
        c for c in df[df["converted"] == 1]["cluster"].unique() if c > -1
    ]

    # Identify non-converters within these clusters
    similar_non_converters = df[
        (df["converted"] == 0) & (df["cluster"].isin(converter_clusters))
    ]

    # Anomaly Detection among non-converters using Isolation Forest
    non_converters_data = df[
        (df["converted"] == 0) & (df["cluster"].isin(converter_clusters))
    ].copy()  # Ensure a copy is made to avoid warnings
    X_non_converters = mlb.transform(non_converters_data["activity_list"])

    # Apply PCA to reduce dimensionality
    X_non_converters_pca = apply_pca(X_non_converters)

    non_converters_data.loc[:, "anomaly"] = anomaly_detection(
        X_non_converters_pca
    )  # Use .loc to avoid warnings

    # Identifying unusual navigation paths
    unusual_non_converters = non_converters_data[non_converters_data["anomaly"] == -1]

    # Selecting top_n similar non-converter sequences
    top_similar_non_converter_seqs = (
        pd.DataFrame(
            [
                [" -> ".join(i[1]), i[0]]
                for i in find_frequent_sequences(
                    similar_non_converters.activity_list.tolist(), min_freq=3, min_len=2
                )
            ],
            columns=["sequence", "freq"],
        )
        .sort_values("freq", ascending=False)
        .head(top_n)
    )

    # Printing head of top similar non-converters
    print("Top Similar Non-Converters Sequences:")
    print_user_and_activity(top_similar_non_converter_seqs)

    # Selecting top_n unusual non-converter sequences
    top_unusual_non_converters_seqs = (
        pd.DataFrame(
            [
                [" -> ".join(i[1]), i[0]]
                for i in find_frequent_sequences(
                    unusual_non_converters.activity_list.tolist(), min_freq=2, min_len=1
                )
            ],
            columns=["sequence", "freq"],
        )
        .sort_values("freq", ascending=False)
        .head(top_n)
    )

    # Printing head of top unusual non-converters
    print("\nTop Unusual Non-Converter Sequences:")
    print_user_and_activity(top_unusual_non_converters_seqs)


def find_frequent_sequences(sequences, min_freq=10, min_len=3):
    """Find frequent sequences using PrefixSpan."""
    ps = PrefixSpan(sequences)
    ps.minlen = min_len
    return ps.frequent(min_freq, closed=True)


def find_divergence_point(seq1, seq2):
    """Find the point of divergence between two sequences."""
    length = max(len(seq1), len(seq2))
    matches = [
        1 if i < len(seq1) and i < len(seq2) and seq1[i] == seq2[i] else 0
        for i in range(length)
    ]
    pct_matches = sum(matches) / length
    for i in range(length):
        if i >= len(seq1) or i >= len(seq2) or seq1[i] != seq2[i]:
            divergence_point = i + 1  # Adding 1 as positions start from 1
            score = ((pct_matches * length) ** divergence_point) / (length**length)
            return i, seq1[i:], seq2[i:], score

    return length, ["<none found>"], ["<none found>"], 0  # No divergence found


def process_pair(non_conv_seq, conv_seq):
    """Process a pair of sequences."""
    idx, non_conv_div, conv_div, divergence_score = find_divergence_point(
        non_conv_seq[1], conv_seq[1]
    )
    if idx != len(non_conv_seq[1]) and non_conv_div and conv_div:
        return {
            "conversion_seq": conv_seq[1],  # Corrected here
            "non_conversion_seq": non_conv_seq[1],  # Corrected here
            "diversion": non_conv_div[0],
            "divergence_score": divergence_score,
        }
    return None


def count_subset_sequences(target_seq, all_sequences):
    """Count how many times a target sequence is a subset of the sequences in a list."""
    count = 0
    for seq in all_sequences:
        if set(target_seq).issubset(set(seq)):
            count += 1
    return count


def analyze_divergence(df, min_freq: int = 10, top_n: int = 10):
    """Analyze divergence points."""

    converters = df[df["converted"] == 1]
    non_converters = df[df["converted"] == 0]

    conv_sequences = find_frequent_sequences(
        converters["activity_list"].tolist(), min_freq=min_freq
    )
    non_conv_sequences = find_frequent_sequences(
        non_converters["activity_list"].tolist(), min_freq=min_freq
    )

    pairs = list(product(non_conv_sequences, conv_sequences))
    results = Parallel(n_jobs=-1)(
        delayed(process_pair)(non_conv_seq, conv_seq)
        for non_conv_seq, conv_seq in tqdm(pairs, desc="Processing pairs")
    )

    results = [
        result for result in results if result is not None
    ]  # Filter out None values
    results_df = pd.DataFrame(results)

    # Prepare a list of all activity_lists to be used for counting
    all_activity_lists = df["activity_list"].tolist()

    # Count total_sessions where the non-conversion sequence is a subset of activity_list
    results_df["pct_users"] = results_df["non_conversion_seq"].apply(
        lambda x: round(
            count_subset_sequences(x, all_activity_lists) / len(non_converters), 2
        )
    )

    # Convert lists to strings to avoid unhashable type error
    results_df["conversion_seq"] = results_df["conversion_seq"].apply(
        lambda x: " -> ".join(x)
    )
    results_df["non_conversion_seq"] = results_df["non_conversion_seq"].apply(
        lambda x: " -> ".join(x)
    )

    # Remove duplicate rows
    results_df = results_df.drop_duplicates()

    results_df["weight"] = results_df["divergence_score"] * results_df["pct_users"]

    results_df.sort_values("weight", ascending=False, inplace=True)

    return results_df
