"""Diversion Analysis."""

from typing import List, Tuple, Union

import pandas as pd
from itertools import product
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from feature import find_frequent_sequences


def find_divergence_point(seq1, seq2) -> Tuple[int, List, List, float]:
    """Find the point of divergence between two sequences and calculate a divergence score.

    The divergence score considers the point of divergence, length of the sequences,
    and the number of differing elements after the divergence point.

    Parameters
    ----------
    seq1 : list
        First sequence.
    seq2 : list
        Second sequence.

    Returns
    -------
    tuple
        A tuple containing the index of divergence, the remaining elements of seq1,
        the remaining elements of seq2, and the divergence score.
    """
    length = min(len(seq1), len(seq2))
    divergence_point = next(
        (i for i, (a, b) in enumerate(zip(seq1, seq2)) if a != b), length
    )
    remaining1 = seq1[divergence_point:]
    remaining2 = seq2[divergence_point:]

    # Calculate the percentage of initial match up to the divergence point.
    initial_match_pct = divergence_point / length

    # Calculate the length of the non-matching tails of both sequences.
    tail_length1 = len(remaining1)
    tail_length2 = len(remaining2)

    # Calculate divergence based on the length of non-matching sequence tails.
    tail_divergence = (tail_length1 + tail_length2) / (
        len(seq1) + len(seq2) - divergence_point
    )

    # The score favors longer initial matches and penalizes longer tails after divergence.
    score = (initial_match_pct**2) / (tail_divergence + 1)

    return divergence_point, remaining1, remaining2, score


def process_pair(non_conv_seq, conv_seq) -> Union[dict, None]:
    """Process a pair of sequences.

    Parameters
    ----------
    non_conv_seq : tuple
        A tuple containing the sequence frequency and non-conversion sequence.
    conv_seq : tuple
        A tuple containing the sequence frequency and conversion sequence.

    Returns
    -------
    dict or None
        A dictionary containing the conversion sequence, non-conversion sequence,
        diversion element, and divergence score, or None if there is no divergence.
    """
    idx, non_conv_div, conv_div, divergence_score = find_divergence_point(
        non_conv_seq[1], conv_seq[1]
    )
    if idx != len(non_conv_seq[1]) and non_conv_div and conv_div:
        return {
            "conversion_seq": conv_seq[1],
            "non_conversion_seq": non_conv_seq[1],
            "diversion": non_conv_div[0] if non_conv_div else None,
            "divergence_score": divergence_score,
            "non_conv_freq": non_conv_seq[0],
        }
    return None


def count_ordered_subset_sequences(target_seq, all_sequences) -> int:
    """Count how many times a target sequence appears in order within the sequences in a list.

    Parameters
    ----------
    target_seq : list
        Target sequence.
    all_sequences : list of list
        List of sequences.

    Returns
    -------
    int
        Count of sequences that contain the target sequence in the same order.
    """

    def is_subsequence(main_seq, sub_seq):
        it = iter(main_seq)
        return all(any(c == ch for c in it) for ch in sub_seq)

    return sum(1 for seq in all_sequences if is_subsequence(seq, target_seq))


def analyze_divergence(
    df, top_n: int = 10, min_freq: int = 10, min_seq_len: int = 3
) -> pd.DataFrame:
    """Analyze divergence points.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with conversion data.
    min_freq : int, optional
        Minimum frequency for sequences to be considered, by default 10.
    top_n : int, optional
        Top N results to be considered, by default 10.
    min_seq_len : int, optional
        Minimum length for sequences to be considered, by default 3.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the analysis results.
    """

    converters = df[df["converted"] == 1]
    non_converters = df[df["converted"] == 0]

    conv_sequences = find_frequent_sequences(
        converters["activity_list"].tolist(), min_freq=min_freq, min_len=min_seq_len
    )
    non_conv_sequences = find_frequent_sequences(
        non_converters["activity_list"].tolist(), min_freq=min_freq, min_len=min_seq_len
    )

    # Restrict conv_sequences and non_conv_sequences to sequences that are not found in both.
    # Data looks like [(freq, seq), (freq, seq), ...]
    # We need to extract the seq part and compare them.
    shared_sequences = set(
        [" -> ".join(seq[1]) for seq in conv_sequences]
    ).intersection(set([" -> ".join(seq[1]) for seq in non_conv_sequences]))

    conv_sequences = [
        seq for seq in conv_sequences if " -> ".join(seq[1]) not in shared_sequences
    ]
    non_conv_sequences = [
        seq for seq in non_conv_sequences if " -> ".join(seq[1]) not in shared_sequences
    ]

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
    # all_activity_lists = df["activity_list"].tolist()
    # len_non_converters = len(non_converters)

    # Convert lists to strings to avoid unhashable type error
    results_df["conversion_seq"] = results_df["conversion_seq"].apply(" -> ".join)
    results_df["non_conversion_seq"] = results_df["non_conversion_seq"].apply(
        " -> ".join
    )

    # Remove duplicate rows
    results_df.drop_duplicates(inplace=True)

    results_df["weight"] = results_df["divergence_score"] * results_df["non_conv_freq"]

    results_df.sort_values("weight", ascending=False, inplace=True)

    return results_df.head(top_n)
