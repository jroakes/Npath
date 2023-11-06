"""Data processing functions for the multi-page analysis."""

from datetime import datetime
import pandas as pd
import numpy as np
from db import query_bigquery


def load_and_clean(
    table_name: str,
    historical_days: int,
    goal_event: str,
    conversion_page_title: str,
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
    conversion_page_title : str
        Title of the conversion page.
    brand : str
        Brand name.

    Returns
    -------
    df : pd.DataFrame
        Cleaned DataFrame.
    """

    df = query_bigquery(table_name, historical_days)
    df = df[df.event_name.isin(["session_start", "page_view", goal_event])].copy()
    df.dropna(subset=["user_pseudo_id", "event_timestamp"], inplace=True)

    df["session_engaged"] = df.session_engaged.fillna(0).astype(int)
    df["user_pseudo_id"] = df.user_pseudo_id.astype(str).str.replace(
        ".", "_", regex=False
    )

    df["event_timestamp_dt"] = pd.to_datetime(
        df.event_timestamp, infer_datetime_format=True, utc=True
    )
    df["event_timestamp"] = (
        df.event_timestamp_dt.view("int64") // 10**9
    )  # converting to UNIX timestamp
    df["event_timestamp_diff"] = (
        df.groupby("user_pseudo_id")["event_timestamp"].diff().fillna(0)
    )

    df.sort_values(
        ["user_pseudo_id", "event_timestamp", "event_name"],
        ascending=[True, True, True],
        inplace=True,
    )

    df.loc[df["event_name"] == "session_start", "event_timestamp"] = (
        df[df["event_name"] == "session_start"]["event_timestamp"] - 1
    )

    fillna_dict = {
        "channel": "(not set)",
        "source": "(not set)",
        "page_location": "/",
        "page_title": "(not set)",
    }
    df.fillna(fillna_dict, inplace=True)

    df["channel"] = df["channel"].str.strip().str.lower()
    df["source"] = df["source"].str.strip().str.lower()

    df["page_location"] = (
        df["page_location"]
        .str.replace(r"(\?|\#).*$", "", regex=True)
        .str.replace(r"^https?\:\/\/[^\/]+", "", regex=True)
        .str.strip()
        .str.lower()
        .str.rstrip("/")
        + "/"
    )

    df["page_title"] = df["page_title"].str.extract(r"(.*?)(?:\s[-|]\s|$)")[0]
    df["page_title"] = (
        df["page_title"].str.replace(brand, "", case=False, regex=True).str.strip()
    )

    df.loc[df.page_location == "/", "page_title"] = "Home"

    conversion_page_title_lower = conversion_page_title.lower().strip()
    df = df[
        ~(
            (df.page_title.str.lower().str.strip() == conversion_page_title_lower)
            & (df.event_name == "page_view")
        )
    ].copy()

    df = df[~((df.page_title.str.len() == 0) & (df.event_name == "page_view"))].copy()
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
        .groupby("user_pseudo_id", as_index=False)
        .agg(
            unique_days=("event_date", "nunique"),
            unique_pages=("page_location", "nunique"),
            pageviews=("event_timestamp", "nunique"),
            engaged_sessions=("session_engaged", "sum"),
            total_sessions=("ga_session_id", "nunique"),
        )
    )

    # Change user_pseudo_id to user_id
    grouped_df.rename(columns={"user_pseudo_id": "user_id"}, inplace=True)
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
    converters = df_in.loc[goal_event_filter, "user_pseudo_id"].unique().tolist()
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
    multi_page_criteria = (df_counts_in.unique_pages < 50) & (
        df_counts_in.unique_pages > 2
    )
    multi_page_users = df_counts_in.loc[multi_page_criteria, "user_id"].tolist()

    user_filter = df_in.user_pseudo_id.isin(multi_page_users)
    filtered_df = df_in.loc[user_filter]

    selected_columns = [
        "user_pseudo_id",
        "page_location",
        "page_title",
        "event_date",
        "event_name",
        "event_timestamp",
        "event_timestamp_diff",
        "geo_region",
        "geo_city",
        "channel",
        "source",
        "ga_session_id",
        "session_engaged",
    ]

    df = (
        filtered_df[selected_columns]
        .sort_values(by="event_timestamp", ascending=True)
        .reset_index(drop=True)
    )

    return df


def process_labels(df_multi_in: pd.DataFrame, goal_event: str) -> pd.DataFrame:
    """
    Assign descriptive labels to events based on event type and other attributes.

    Parameters
    ----------
    df_multi_in : pd.DataFrame
        Input DataFrame containing event data.
    goal_event : str
        Name of the goal event.

    Returns
    -------
    df : pd.DataFrame
        Output DataFrame with a new 'label' column describing each event.
    """

    df = df_multi_in.copy()

    lookup = {
        "(direct)": "Navigated directly to our website",
        "(organic)": "Navigated via an online search",
        "(referral)": "Navigated via a link from unknown source",
        "(not set)": "Navigated via a link from unknown source",
    }

    page_view_filter = df.event_name == "page_view"
    session_start_filter = df.event_name == "session_start"

    df.loc[page_view_filter, "label"] = "Navigated to: " + df.loc[
        page_view_filter, "page_title"
    ].astype(str)

    df.loc[session_start_filter, "label"] = (
        df.loc[session_start_filter, "channel"]
        .map(lookup)
        .fillna("Found us via marketing efforts")
    )

    session_start_criteria = (
        session_start_filter
        & (df.label == "Navigated via a link from unknown source")
        & (df.source != "(not set)")
    )

    df.loc[session_start_criteria, "label"] = "Navigated via a link from " + df.loc[
        session_start_criteria, "source"
    ].astype(str)

    df["label"].fillna("", inplace=True)

    return df


def process_aggregation(df_multi_in: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates user data into a summary format.

    Parameters
    ----------
    df_multi_in : pd.DataFrame
        Input DataFrame with event data for multiple users.

    Returns
    -------
    df : pd.DataFrame
        Output DataFrame with aggregated data per user.
    """

    def oset(x):
        """Order preserving set function."""
        return list(dict.fromkeys(x))

    # Aggregating data per user
    df = df_multi_in.groupby("user_pseudo_id", as_index=False).agg(
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

    return df


def get_data(
    table_name: str,
    historical_days: int,
    goal_event: str,
    conversion_page_title: str,
    brand: str,
) -> pd.DataFrame:
    """
    Fetch and process data for analysis.

    Parameters
    ----------
    table_name : str
        Name of the BigQuery table to query.
    historical_days : int
        Number of days to query from BigQuery.
    goal_event : str
        Name of the goal event.
    conversion_page_title : str
        Title of the conversion page.
    brand : str
        Brand name.

    Returns
    -------
    df_multi_agg : pd.DataFrame
        Aggregated DataFrame with user activity and conversion data.
    """

    df = load_and_clean(
        table_name, historical_days, goal_event, conversion_page_title, brand
    )
    df_counts = process_counts(df)
    df_multi = process_multi(df, df_counts)
    df_multi = process_labels(df_multi, goal_event)
    df_multi_agg = process_aggregation(df_multi)

    # Get converters
    converters_set = set(process_converters(df, goal_event))

    # Adding converters to df_multi_agg as 0/1
    df_multi_agg["converted"] = df_multi_agg.user_id.isin(converters_set).astype(int)

    return df_multi_agg


# Path: data.py
