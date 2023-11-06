"""Conversion Analysis."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import umap
import hdbscan
from sklearn.ensemble import IsolationForest

from feature import find_frequent_sequences


def standardize_data(X: np.ndarray) -> np.ndarray:
    """
    Standardizes the data by removing the mean and scaling to unit variance.

    Parameters
    ----------
    X : np.ndarray
        The input data to be standardized.

    Returns
    -------
    np.ndarray
        The standardized version of the input data.
    """
    scaler = StandardScaler(with_mean=True, with_std=True, copy=False)
    return scaler.fit_transform(X)


def apply_umap(X: np.ndarray, n_components: int = 5) -> np.ndarray:
    """
    Applies Uniform Manifold Approximation and Projection (UMAP) to reduce dimensionality.

    Parameters
    ----------
    X : np.ndarray
        The input data to be transformed.
    n_components : int, optional
        Number of components to keep, by default 5.

    Returns
    -------
    np.ndarray
        The transformed data.
    """
    reducer = umap.UMAP(n_components=n_components)
    return reducer.fit_transform(X)


def apply_pca(X: np.ndarray, n_components: int = 5) -> np.ndarray:
    """
    Applies Principal Component Analysis (PCA) to reduce dimensionality.

    Parameters
    ----------
    X : np.ndarray
        The input data to be transformed.
    n_components : int, optional
        Number of components to keep, by default 5.

    Returns
    -------
    np.ndarray
        The transformed data.
    """
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X)


def perform_clustering(X: np.ndarray, min_cluster_size: int = 3) -> np.ndarray:
    """
    Performs clustering using HDBSCAN.

    Parameters
    ----------
    X : np.ndarray
        The input data to be clustered.
    min_cluster_size : int, optional
        The minimum size of clusters, by default 3.

    Returns
    -------
    np.ndarray
        The labels of the clusters for each data point.
    """
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, gen_min_span_tree=True
    )
    labels = clusterer.fit_predict(X)
    return labels, clusterer


def anomaly_detection(X: np.ndarray, contamination: float = 0.05) -> np.ndarray:
    """
    Performs anomaly detection using Isolation Forest.

    Parameters
    ----------
    X : np.ndarray
        The input data for anomaly detection.
    contamination : float, optional
        The amount of contamination of the data set, i.e., the proportion
        of outliers in the data set, by default 0.05.

    Returns
    -------
    np.ndarray
        The anomaly labels for each data point.
    """
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    return iso_forest.fit_predict(X)


def print_user_and_activity(df: pd.DataFrame, top_n: int = 5) -> None:
    """
    Prints user and activity.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing user and activity data.
    top_n : int, optional
        The number of top activities to print, by default 5.

    Returns
    -------
    None
    """
    for index, (idx, row) in enumerate(df.iterrows(), start=1):
        print(f"{index}. Count: {row['freq']}, Sequence: {row['sequence']}\n")
        if index == top_n:
            break


def convertor_review(df: pd.DataFrame, top_n: int = 10, reducer: str = "umap") -> None:
    """Review converters and non-converters.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing user activity data.
    top_n : int, optional
        Number of top records to display, by default 10.
    reducer : str, optional
        Dimensionality reduction technique to use, by default "umap". Options are "umap" and "pca".

    Returns
    -------
    None
    """
    # Convert activity_list to binary matrix
    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform(df["activity_list"])

    # Impute any missing values in the binary matrix
    imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
    X_imputed = imputer.fit_transform(X)

    # Standardize the data
    X_standardized = standardize_data(X_imputed)

    # Apply PCA to reduce dimensionality
    if reducer == "pca":
        Xm = apply_pca(X_standardized)
    else:
        Xm = apply_umap(X_standardized)

    # Clustering using HDBSCAN
    cluster_labels, _ = perform_clustering(Xm)
    df["cluster"] = cluster_labels

    # Identify clusters that have converters (-1 is noise)
    converter_clusters = [
        c for c in df[df["converted"] == 1]["cluster"].unique() if c > -1
    ]

    # Identify non-converters within these clusters
    similar_non_converters = df[
        (df["converted"] == 0) & (df["cluster"].isin(converter_clusters))
    ].copy()

    # Anomaly Detection among non-converters using Isolation Forest
    X_non_converters = mlb.transform(similar_non_converters["activity_list"])
    X_non_converters_standardized = standardize_data(X_non_converters)

    if reducer == "pca":
        X_non_converters_pca = apply_pca(X_non_converters_standardized)
    else:
        X_non_converters_pca = apply_umap(X_non_converters_standardized)

    similar_non_converters.loc[:, "anomaly"] = anomaly_detection(X_non_converters_pca)

    # Selecting top_n similar non-converter sequences
    top_similar_non_converter_seqs = (
        pd.DataFrame(
            [
                [" -> ".join(i[1]), i[0]]
                for i in find_frequent_sequences(
                    similar_non_converters.activity_list.tolist(), min_freq=2
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

    # Identifying unusual navigation paths
    unusual_non_converters = similar_non_converters[
        similar_non_converters["anomaly"] == -1
    ]

    print(f"Number of unusual non-converters: {len(unusual_non_converters)}")

    # Checking if unusual_non_converters is not empty
    if not unusual_non_converters.empty:
        # Selecting top_n unusual non-converter sequences
        top_unusual_non_converters_seqs = (
            pd.DataFrame(
                [
                    [" -> ".join(i[1]), i[0]]
                    for i in find_frequent_sequences(
                        unusual_non_converters.activity_list.tolist()
                    )
                ],
                columns=["sequence", "freq"],
            )
            .sort_values("freq", ascending=False)
            .head(top_n)
        )

        # If there are no frequent sequences found
        if top_unusual_non_converters_seqs.empty:
            print("No frequent sequences found among unusual non-converters.")
        else:
            # Printing head of top unusual non-converters
            print("\nTop Unusual Non-Converter Sequences:")
            print_user_and_activity(top_unusual_non_converters_seqs)
    else:
        print("There are no unusual non-converter sequences detected.")
