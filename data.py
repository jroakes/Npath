import re
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
):
    df = query_bigquery(table_name, historical_days)

    df = df[df.event_name.isin(["session_start", "page_view", goal_event])].copy()

    df["session_engaged"] = df.session_engaged.fillna(0).astype(int)
    df.dropna(subset=["user_pseudo_id", "event_timestamp"], inplace=True)
    df["user_pseudo_id"] = df.user_pseudo_id.map(lambda x: x.replace(".", "_"))
    df["event_timestamp_dt"] = pd.to_datetime(
        df.event_timestamp, infer_datetime_format=True, utc=True
    )
    df["event_timestamp"] = df.event_timestamp_dt.map(datetime.timestamp)
    df["event_timestamp_diff"] = 0

    # Sort the DataFrame by 'user_pseudo_id', 'event_timestamp' and 'event_name' in ascending order
    df.sort_values(
        ["user_pseudo_id", "event_timestamp", "event_name"],
        ascending=[True, True, True],
        inplace=True,
    )

    # Calculate the difference in seconds for 'event_timestamp' within each 'user_pseudo_id' group, but only for rows where event_name is 'page_view'
    df.loc[df["event_name"] == "page_view", "event_timestamp_diff"] = (
        df[df["event_name"] == "page_view"]
        .groupby("user_pseudo_id")["event_timestamp"]
        .diff()
    )

    # For the first row of each 'user_pseudo_id' for 'page_view', the difference would be NaN after applying diff(), fill it with 0.
    df["event_timestamp_diff"].fillna(0, inplace=True)

    # Minus one from session_start timestamps to make sure they are always in front of pageviews
    df.loc[df["event_name"] == "session_start", "event_timestamp"] = (
        df[df["event_name"] == "session_start"]["event_timestamp"] - 1
    )

    df.channel.fillna("(not set)", inplace=True)
    df.source.fillna("(not set)", inplace=True)
    df.page_location.fillna("/", inplace=True)

    # Normalize URLs and channels
    df["channel"] = df.channel.map(lambda x: x.strip().lower())
    df["source"] = df.source.map(lambda x: x.strip().lower())
    df["page_location"] = df.page_location.map(
        lambda x: re.sub("(\?|\#).*$", "", x).strip().lower()
    )
    df["page_location"] = df.page_location.map(
        lambda x: re.sub("^https?\:\/\/[^\/]+", "", x).strip().lower()
    )
    df["page_location"] = df.page_location.map(lambda x: x if x[-1] == "/" else x + "/")
    df["page_title"] = df.page_title.map(lambda x: re.split(r"\s[-|]\s", x)[0])
    df["page_title"] = df["page_title"].map(
        lambda x: re.sub(brand, "", x, flags=re.IGNORECASE).strip()
    )

    df.loc[df.page_location == "/", "page_title"] = "Home"

    # Remove rows where page_title matches the conversion_page_title. Lowercase both strings first.
    df = df[
        ~(
            (
                df.page_title.str.lower().str.strip()
                == conversion_page_title.lower().strip()
            )
            & (df.event_name == "page_view")
        )
    ].copy()

    # Get rid of empty titles
    df = df[~((df.page_title.str.len() == 0) & (df.event_name == "page_view"))].copy()

    # Double sort to make sure that `session_start` is first
    df.sort_values("event_timestamp", ascending=True, inplace=True)

    return df


def process_counts(df_in):
    df = df_in[df_in.event_name == "page_view"].copy()
    df = df.groupby(["user_pseudo_id"], as_index=False).agg(
        {
            "event_date": "nunique",
            "page_location": "nunique",
            "event_timestamp": "nunique",
            "session_engaged": "sum",
            "ga_session_id": "nunique",
        }
    )
    df.columns = [
        "user_id",
        "unique_days",
        "unique_pages",
        "pageviews",
        "engaged_sessions",
        "total_sessions",
    ]
    df.engaged_sessions = round(df.engaged_sessions / df.pageviews, 2)
    df.sort_values(by="pageviews", ascending=False, inplace=True)

    return df


def process_converters(df_in, goal_event) -> list:
    df = df_in[df_in.event_name == goal_event].copy()
    converters = df.user_pseudo_id.unique().tolist()

    return converters


def process_multi(df_in, df_counts_in):
    multi_page_users = df_counts_in[
        (df_counts_in.unique_pages < 50) & (df_counts_in.unique_pages > 2)
    ].user_id.tolist()

    df = df_in[df_in.user_pseudo_id.isin(multi_page_users)].copy()

    df = df[
        [
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
    ].copy()

    df.sort_values(by="event_timestamp", ascending=True, inplace=True)
    df.reset_index(inplace=True, drop=True)

    return df


def process_labels(df_multi_in: pd.DataFrame, goal_event: str):
    df = df_multi_in.copy()

    lookup = {
        "(direct)": "Navigated directly to our website",
        "(organic)": "Navigated via an online search",
        "(referral)": "Navigated via a link from unknown source",
        "(not set)": "Navigated via a link from unknown source",
    }

    df.loc[df.event_name == "page_view", "label"] = df[(df.event_name == "page_view")][
        "page_title"
    ].map(lambda x: f"Navigated to: {x}")
    df.loc[df.event_name == "session_start", "label"] = df[
        (df.event_name == "session_start")
    ]["channel"].map(lambda x: lookup.get(x, "Found us via marketing efforts"))

    # fill in referrer with "Navigated via a link from (source)"
    df.loc[
        (df.event_name == "session_start")
        & (df.label == "Navigated via a link from unknown source")
        & (df.source != "(not set)"),
        "label",
    ] = df[
        (df.event_name == "session_start")
        & (df.label == "Navigated via a link from unknown source")
        & (df.source != "(not set)")
    ][
        "source"
    ].map(
        lambda x: f"Navigated via a link from {x}"
    )

    df["label"] = df["label"].fillna("")

    return df


def process_aggregation(df_multi_in: pd.DataFrame):
    def oset(x):
        return [i for i in list(dict.fromkeys(x).keys()) if len(i) > 0]

    df_multi = df_multi_in.copy()

    df = df_multi.groupby(["user_pseudo_id"], as_index=False).agg(
        {
            "event_date": "nunique",
            "page_location": "nunique",
            "event_timestamp": "nunique",
            "session_engaged": "sum",
            "ga_session_id": "nunique",
            "label": lambda x: [i for i in oset(x)],
        }
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

    df["activity_list_text"] = df["activity_list"].map(lambda x: " -> ".join(x))

    # engaged sessions is correct becuse session_start and first page_view should have the same timestamp.
    df.engaged_sessions = df.engaged_sessions / df.unique_pageviews
    df.sort_values(by="unique_pageviews", ascending=False, inplace=True)
    df.reset_index(inplace=True, drop=True)

    return df


def get_data(
    table_name: str,
    historical_days: int,
    goal_event: str,
    conversion_page_title: str,
    brand: str,
) -> pd.DataFrame:
    df = load_and_clean(
        table_name, historical_days, goal_event, conversion_page_title, brand
    )
    df_counts = process_counts(df)
    df_multi = process_multi(df, df_counts)
    df_multi = process_labels(df_multi, goal_event)
    df_multi_agg = process_aggregation(df_multi)

    # Get converters
    converters = process_converters(df, goal_event)

    # add converters to df_multi_agg as 0/1
    df_multi_agg["converted"] = df_multi_agg.user_id.map(
        lambda x: 1 if x in converters else 0
    )

    return df_multi_agg
