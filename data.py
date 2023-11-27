"""Data processing functions for the multi-page analysis."""

import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

from db import query_bigquery


def load_and_clean(
    table_name: str,
    historical_days: int,
    goal_event: str,
    brand: str,
) -> pd.DataFrame:
    """
    Load data from BigQuery and clean it up for further processing.

    Parameters
    ----------
    table_name : str
        Name of the BigQuery table to query.
    historical_days : int
        Number of days to query from BigQuery.
    goal_event : str
        Name of the goal event.
    conversion_page_path : list
        URL path of the conversion page (list).
    brand : list
        Brand name (list).

    Returns
    -------
    df : pd.DataFrame
        Cleaned DataFrame.
    """

    df = query_bigquery(table_name, historical_days)

    # Reduce to relevant events
    df = df[df.event_name.isin(["session_start", "page_view", goal_event])].copy()

    # Remove rows with null values in important columns
    df.dropna(subset=["user_pseudo_id", "event_timestamp", "event_name"], inplace=True)

    # Clean up data types
    df["session_engaged"] = df.session_engaged.fillna(0).astype(int)
    df["user_pseudo_id"] = df.user_pseudo_id.astype(str).str.replace(
        ".", "_", regex=False
    )

    # Remove rows with ___PII_REDACTED___ for user_pseudo_id
    df = df[~df.user_pseudo_id.str.contains("___PII_REDACTED___")]

    # Duplicate event_name == 'session_start' rows and rename event_name to 'page_view'
    # This is supposed to ensure that the first page_view is not missing
    df_session_start = df[
        (df.event_name == "session_start") & (df.page_location.notnull())
    ].copy()
    df_session_start["event_name"] = "page_view"
    df = pd.concat([df, df_session_start], ignore_index=True)

    # Drop rows where event_name == 'page_view' and page_location is null
    # We don't need the page_view event if we don't know the page_location
    df = df[~((df.event_name == "page_view") & (df.page_location.isnull()))]

    # Cast event_timestamp to datetime
    df["event_timestamp"] = pd.to_datetime(
        df.event_timestamp, infer_datetime_format=True, errors="coerce", utc=True
    )

    session_starts = df["event_name"] == "session_start"
    duplicated_events = df.duplicated(
        subset=["user_pseudo_id", "event_timestamp"], keep=False
    )
    needs_adjustment = session_starts & duplicated_events

    # Adjust timestamps where necessary
    df.loc[needs_adjustment, "event_timestamp"] -= pd.Timedelta(seconds=2)

    # Convert event_timestamp to UNIX timestamp (seconds)
    df["event_timestamp_sec"] = (
        df.event_timestamp.view("int64") // 10**9
    )  # converting to UNIX timestamp

    # Sort by event_timestamp in ascending order
    df.sort_values("event_timestamp", ascending=True, inplace=True)

    # Calculate the time difference between events for each user
    df["event_timestamp_diff"] = (
        df.groupby("user_pseudo_id")["event_timestamp_sec"].diff().fillna(0)
    )

    # Clean up and standardize the rest of the data
    fillna_dict = {
        "campaign": "(not set)",
        "medium": "(not set)",
        "source_medium": "(not set)",
        "source": "(not set)",
        "page_location": "/",
        "page_title": "(not set)",
    }
    df.fillna(fillna_dict, inplace=True)

    df["campaign"] = df["campaign"].str.strip().str.lower()
    df["source"] = df["source"].str.strip().str.lower()
    df["medium"] = df["medium"].str.strip().str.lower()
    df["source_medium"] = df["source_medium"].str.strip().str.lower()

    # Update medium "(not set)" to be direct
    df.loc[df["medium"] == "(none)", "medium"] = "direct"
    df.loc[df["source_medium"] == "	(direct) / (none)", "medium"] = "direct"

    # Change user_pseudo_id to user_id
    df.rename(columns={"user_pseudo_id": "user_id"}, inplace=True)

    df["page_location"] = (
        df["page_location"]
        .str.replace(r"(\?|\#).*$", "", regex=True)
        .str.replace(r"^https?\:\/\/[^\/]+", "", regex=True)
        .str.strip()
        .str.lower()
        .str.rstrip("/")
        + "/"
    )

    df["page_title"] = df["page_title"].fillna("")

    # Extract part of Title bevor - or | or end of string
    df["page_title"] = df["page_title"].str.extract(r"(.*?)(?:\s[-|]\s|$)")[0]

    # Remove brand name from page_title
    brand_regex = "|".join([b.strip().lower() for b in brand])
    df["page_title"] = (
        df["page_title"]
        .str.replace(brand_regex, "", case=False, regex=True)
        .str.strip()
    )

    # Replace empty page_title with (not set)
    df.loc[df.page_title == "", "page_title"] = "(not set)"

    # Replace / path page_title with Home
    df.loc[df.page_location == "/", "page_title"] = "Home"

    df.sort_values("event_timestamp", ascending=True, inplace=True)

    return df


def process_counts(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Process counts for each user.

    Parameters
    ----------
    df_in : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    df : pd.DataFrame
        Output DataFrame with aggregated counts for each user.
    """
    page_view_filter = df_in.event_name == "page_view"
    grouped_df = (
        df_in.loc[page_view_filter]
        .groupby("user_id", as_index=False)
        .agg(
            unique_days=("event_date", "nunique"),
            unique_pages=("page_location", "nunique"),
            pageviews=("event_timestamp", "nunique"),
            engaged_sessions=("session_engaged", "sum"),
            total_sessions=("ga_session_id", "nunique"),
        )
    )

    engaged_sessions_ratio = grouped_df["engaged_sessions"] / grouped_df["pageviews"]
    grouped_df["engaged_sessions"] = engaged_sessions_ratio.round(2)
    grouped_df.sort_values(by="pageviews", ascending=False, inplace=True)
    grouped_df.reset_index(drop=True, inplace=True)

    return grouped_df


def process_converters(df_in: pd.DataFrame, goal_event: str) -> list:
    """
    Extract unique user ids for specified goal events.

    Parameters
    ----------
    df_in : pd.DataFrame
        Input DataFrame.
    goal_event : str
        Name of the goal event.

    Returns
    -------
    converters : list
        List of unique user ids who achieved the goal event.
    """
    goal_event_filter = df_in.event_name == goal_event
    converters = df_in.loc[goal_event_filter, "user_id"].unique().tolist()
    return converters


def process_multi(df_in: pd.DataFrame, df_counts_in: pd.DataFrame) -> pd.DataFrame:
    """
    Filters and processes data for users who have viewed multiple pages.

    Parameters
    ----------
    df_in : pd.DataFrame
        Input DataFrame.
    df_counts_in : pd.DataFrame
        Input DataFrame with counts.

    Returns
    -------
    df : pd.DataFrame
        Output DataFrame with selected columns sorted by event timestamp.
    """
    multi_page_criteria = (df_counts_in.unique_pages < 75) & (
        df_counts_in.unique_pages > 2
    )
    multi_page_users = df_counts_in.loc[multi_page_criteria, "user_id"].tolist()

    user_filter = df_in.user_id.isin(multi_page_users)
    filtered_df = df_in.loc[user_filter]

    selected_columns = [
        "user_id",
        "page_location",
        "page_title",
        "event_date",
        "event_name",
        "event_timestamp",
        "event_timestamp_sec",
        "event_timestamp_diff",
        "geo_region",
        "geo_city",
        "device_category",
        "campaign",
        "source",
        "medium",
        "source_medium",
        "ga_session_id",
        "session_engaged",
    ]

    df = (
        filtered_df[selected_columns]
        .sort_values(by="event_timestamp", ascending=True)
        .reset_index(drop=True)
    )

    return df


def process_conversion_path(
    df_multi_in: pd.DataFrame, conversion_page_path: list
) -> pd.DataFrame:
    """Removes rows from the DataFrame where the user has converted.

    Parameters
    ----------
    df_multi_in : pd.DataFrame
        Input DataFrame with event data for multiple users.
    conversion_page_path : list
        URL path of the conversion page (list).

    Returns
    -------
    df : pd.DataFrame
        Output DataFrame with rows removed where the user has converted.
    """

    conversion_page_path = [p.strip() for p in conversion_page_path]

    # Sort by event_timestamp in ascending order
    df_multi_in.sort_values(by="event_timestamp", ascending=True, inplace=True)

    # Create a mask for rows where the page_location is in the conversion_page_path
    mask = df_multi_in.page_location.isin(conversion_page_path)

    # Get the timestamps for the first conversion_page_path page_location per user_id
    first_conversion_timestamps = (
        df_multi_in[mask]
        .groupby("user_id")["event_timestamp"]
        .first()
        .reset_index(name="first_conversion_timestamp")
    )

    # Filter to rows where the event_timestamp is before the first_conversion_timestamp
    # for that user_id
    df = df_multi_in.merge(
        first_conversion_timestamps, how="left", on="user_id", validate="m:1"
    )

    # Fill missing first_conversion_timestamps with value from 2099
    # df["event_timestamp"] = pd.to_datetime(
    #    df.event_timestamp, infer_datetime_format=True, errors="coerce", utc=True
    # )
    df["first_conversion_timestamp"] = df["first_conversion_timestamp"].fillna(
        pd.Timestamp("2099-01-01 00:00:00", tz="UTC")
    )

    df = df[(df.event_timestamp < df.first_conversion_timestamp)].copy()

    # Drop the first_conversion_timestamp column
    df.drop(columns=["first_conversion_timestamp", "event_timestamp_sec"], inplace=True)

    return df


def process_removals(df_multi_in: pd.DataFrame, removal_phrases: list) -> pd.DataFrame:
    """Removes rows for user_ids that have visited a page with a removal phrase.

    Process:
    1. Filter to rows where the page_location contains a removal phrase.
    2. Grab the unique user_ids for those rows.
    3. Filter to rows where the user_id is not in the list of unique user_ids.

    Parameters
    ----------
    df_multi_in : pd.DataFrame
        Input DataFrame with event data for multiple users.
    removal_phrases : list
        List of phrases to remove.

    Returns
    -------
    df : pd.DataFrame
        Output DataFrame with rows removed where the user has converted.
    """

    df = df_multi_in.copy()

    # Filter to rows where the page_location contains a removal phrase
    mask = df.page_location.str.contains(
        rf"\b(?:{'|'.join(removal_phrases)})\b", case=False, regex=True
    )

    # Grab the unique user_ids for those rows
    user_ids = df.loc[mask, "user_id"].unique().tolist()

    # Filter to rows where the user_id is not in the list of unique user_ids
    return df[~df.user_id.isin(user_ids)]


def process_medium_mix(df_in: pd.DataFrame, strategy: str = "first") -> pd.DataFrame:
    """Attribute credit to mediums based on a scoring strategy.

    Parameters
    ----------
    df_in : pd.DataFrame
        Input DataFrame with event data for multiple users.
    strategy : str
        Scoring strategy to use.
        Options are 'first', 'even' and 'last'.

    Returns
    -------
    scores_df : pd.DataFrame
        Output DataFrame with scores for each medium.
    """

    if strategy not in ["first", "even", "last"]:
        raise ValueError("strategy must be either 'first', 'even' or 'last'")

    df = df_in[df_in.event_name == "session_start"].copy()

    # Group by 'medium' and compute frequency
    medium_df = df.groupby("source_medium").size().reset_index(name="count")
    medium_df = medium_df[medium_df["count"] > medium_df["count"].quantile(0.10)]

    # Preprocess medium names
    medium_df["medium_processed"] = medium_df["source_medium"].apply(
        lambda x: f"medium_{re.sub(r'[^a-zA-Z0-9]+', '_', x).strip(' _')}"
    )
    medium_dict = dict(zip(medium_df["source_medium"], medium_df["medium_processed"]))

    # Replace mediums not in medium_dict with "(none)"
    df["source_medium"] = df["source_medium"].where(
        df["source_medium"].isin(medium_dict), "(not set)"
    )

    # Convert 'medium' to a list per 'user_id'
    user_mediums = df.groupby("user_id")["source_medium"].apply(list)

    # Initialize MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    medium_matrix = mlb.fit_transform(user_mediums)

    # Apply scoring strategy
    if strategy == "first":
        scores = (medium_matrix.cumsum(axis=1) == 1) & (medium_matrix == 1)
    elif strategy == "even":
        # Count unique mediums for each user
        unique_medium_counts = np.array([len(set(mediums)) for mediums in user_mediums])
        scores = np.divide(medium_matrix, unique_medium_counts[:, np.newaxis])
    elif strategy == "last":
        reversed_cumsum = np.fliplr(np.fliplr(medium_matrix).cumsum(axis=1))
        scores = (reversed_cumsum == 1) & (medium_matrix == 1)

    # Convert scores to float for homogeneity
    scores = scores.astype(float)

    # Convert scores to DataFrame and set column names
    scores_df = pd.DataFrame(scores, columns=mlb.classes_)
    scores_df = scores_df.rename(
        columns={old: medium_dict.get(old, old) for old in scores_df.columns}
    )

    # Add 'user_id' to the scores DataFrame
    scores_df["user_id"] = user_mediums.index

    return scores_df


def process_labels(
    df_multi_in: pd.DataFrame, page_repr: str = "page_location"
) -> pd.DataFrame:
    """
    Assign descriptive labels to events based on event type and other attributes.

    Parameters
    ----------
    df_multi_in : pd.DataFrame
        Input DataFrame containing event data.
    page_repr : str
        Name of the column to use for page representation.
        Options are 'page_location' and 'page_title'.

    Returns
    -------
    df : pd.DataFrame
        Output DataFrame with a new 'label' column describing each event.
    """

    if page_repr not in ["page_location", "page_title"]:
        raise ValueError("page_repr must be either 'page_location' or 'page_title'")

    df = df_multi_in.copy()

    page_view_filter = df.event_name == "page_view"
    session_start_filter = df.event_name == "session_start"

    df.loc[page_view_filter, "label"] = "Navigated to: " + df.loc[
        page_view_filter, page_repr
    ].astype(str)

    df.loc[session_start_filter, "label"] = "Originated from: " + df.loc[
        session_start_filter, "source_medium"
    ].astype(str).fillna("Originated from: (not set)")

    unknown_source_criteria = (
        session_start_filter
        & (df.label == "Originated from: (not set)")
        & (df.source != "(not set)")
    )

    df.loc[unknown_source_criteria, "label"] = "Originated from: " + df.loc[
        unknown_source_criteria, "source"
    ].astype(str)

    df["label"].fillna("", inplace=True)

    return df


def process_aggregation(
    df_multi_in: pd.DataFrame, df_attr: pd.DataFrame
) -> pd.DataFrame:
    """
    Aggregates user data into a summary format.

    Parameters
    ----------
    df_multi_in : pd.DataFrame
        Input DataFrame with event data for multiple users.
    df_attr : pd.DataFrame
        Input DataFrame with scores for each medium.

    Returns
    -------
    df : pd.DataFrame
        Output DataFrame with aggregated data per user.
    """

    def oset(x):
        """Order preserving set function."""
        return list(dict.fromkeys(x))

    # Aggregating data per user
    df = df_multi_in.groupby("user_id", as_index=False).agg(
        event_date=("event_date", "nunique"),
        page_location=("page_location", "nunique"),
        event_timestamp=("event_timestamp", "nunique"),
        session_engaged=("session_engaged", "sum"),
        ga_session_id=("ga_session_id", "nunique"),
        activity_list=("label", lambda x: [i for i in oset(x) if i]),
    )

    df.columns = [
        "user_id",
        "unique_days",
        "unique_pages",
        "unique_pageviews",
        "engaged_sessions",
        "total_sessions",
        "activity_list",
    ]

    df["activity_list_text"] = df["activity_list"].map(" -> ".join)

    # Correcting engaged_sessions value
    df["engaged_sessions"] = (
        df["engaged_sessions"].div(df["unique_pageviews"]).replace([np.inf, -np.inf], 0)
    )

    # Sorting by unique_pageviews in descending order
    df.sort_values("unique_pageviews", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Adding medium scores
    df = df.merge(df_attr, how="left", on="user_id")

    return df


import time
import pandas as pd


def get_data(
    table_name: str,
    historical_days: int,
    goal_event: str,
    conversion_page_path: str,
    brand: str,
    removal_phrases: list = ["job", "job", "career", "careers", "apply"],
    page_repr: str = "page_location",
    strategy: str = "first",
) -> pd.DataFrame:
    """
    Fetch and process data for analysis.
    ...

    Returns
    -------
    df_multi : pd.DataFrame
        DataFrame with user activity.
    df_multi_agg : pd.DataFrame
        Aggregated DataFrame with user activity and conversion data.
    """

    start_time = time.time()
    df = load_and_clean(table_name, historical_days, goal_event, brand)
    print(f"load_and_clean took {(time.time() - start_time) / 60:.2f} minutes")

    start_time = time.time()
    df_counts = process_counts(df)
    print(f"process_counts took {(time.time() - start_time) / 60:.2f} minutes")

    start_time = time.time()
    df_multi = process_multi(df, df_counts)
    print(f"process_multi took {(time.time() - start_time) / 60:.2f} minutes")

    start_time = time.time()
    df_multi = process_removals(df_multi, removal_phrases)
    print(f"process_removals took {(time.time() - start_time) / 60:.2f} minutes")

    start_time = time.time()
    df_attr = process_medium_mix(df, strategy)
    print(f"process_medium_mix took {(time.time() - start_time) / 60:.2f} minutes")

    start_time = time.time()
    df_multi = process_conversion_path(df_multi, conversion_page_path)
    print(f"process_conversion_path took {(time.time() - start_time) / 60:.2f} minutes")

    start_time = time.time()
    df_multi = process_labels(df_multi, page_repr)
    print(f"process_labels took {(time.time() - start_time) / 60:.2f} minutes")

    start_time = time.time()
    df_multi_agg = process_aggregation(df_multi, df_attr)
    print(f"process_aggregation took {(time.time() - start_time) / 60:.2f} minutes")

    start_time = time.time()
    converters_set = set(process_converters(df, goal_event))
    print(f"process_converters took {(time.time() - start_time) / 60:.2f} minutes")

    start_time = time.time()
    df_multi_agg["converted"] = df_multi_agg.user_id.isin(converters_set).astype(int)
    df_multi["converted"] = df_multi.user_id.isin(converters_set).astype(int)
    print(f"Setting 'converted' took {(time.time() - start_time) / 60:.2f} minutes")

    return df_multi, df_multi_agg


# Path: data.py
