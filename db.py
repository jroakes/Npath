"""This module contains functions for querying data from BigQuery."""

import re
import os
from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd

PROJECT_ID = "locomotive-ga4-data"
# Querying the data from BigQuery
QUERY_TEMPLATE = """WITH

raw_ga_4 AS (

            SELECT
            * except(row)
            FROM (
            SELECT
                -- extracts date from source table
                parse_date('%Y%m%d',regexp_extract(_table_suffix,'[0-9]+')) as table_date,
                -- flag to indicate if source table is `events_intraday_`
                case when _table_suffix like '%intraday%' then true else false end as is_intraday,
                *,
                row_number() over (partition by user_pseudo_id, event_name, event_timestamp order by event_timestamp) as row
            FROM
                `{table_name}`

            WHERE PARSE_DATE('%Y%m%d', _table_suffix) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL {historical_days} DAY) AND CURRENT_DATE()

                )
            WHERE
            row = 1

            ),

events AS (
    SELECT
        parse_date('%Y%m%d', max(event_date)) event_date,
        event_name,
        timestamp_micros(event_timestamp) as event_timestamp,
        user_pseudo_id,

        COALESCE(NULLIF(MAX(geo.region), ''), '(not set)') AS geo_region,
        COALESCE(NULLIF(MAX(geo.city), ''), '(not set)') AS geo_city,
        COALESCE(NULLIF(MAX(device.category), ''), '(not set)') AS device_category,

        concat(user_pseudo_id, '.',max(if(params.key = 'ga_session_id', params.value.int_value, null))) ga_session_id,
        max(if(params.key = 'ga_session_number', params.value.int_value, null)) ga_session_number,
        ifnull(cast(max(if(params.key = 'session_engaged', params.value.string_value, null)) as int64), 0) session_engaged,
        max(if(params.key = 'page_location', params.value.string_value, null)) page_location,
        max(if(params.key = 'page_title', params.value.string_value, null)) page_title,

        -- If there are no values for the traffic, we default to inital user values.  collected_traffic_source is new in June 2023.
        -- This gets close to GA4 Actual data.
        ifnull(ifnull(max(collected_traffic_source.manual_campaign_name), max(traffic_source.name)), '(not set)') campaign,
        ifnull(ifnull(max(collected_traffic_source.manual_medium), max(traffic_source.medium)), '(not set)') medium,
        ifnull(ifnull(max(collected_traffic_source.manual_source), max(traffic_source.source)), '(not set)') source,

        concat(ifnull(ifnull(max(collected_traffic_source.manual_source), max(traffic_source.source)), '(direct)'), " / ", ifnull(ifnull(max(collected_traffic_source.manual_medium), max(traffic_source.medium)), '(not set)')) source_medium

        FROM raw_ga_4,
        UNNEST(event_params) AS params
        WHERE user_pseudo_id IS NOT NULL
        GROUP BY user_pseudo_id, event_timestamp, event_name
        )



SELECT
*
FROM events
ORDER BY event_timestamp ASC
"""


def query_bigquery(table_name: str, historical_days: int) -> pd.DataFrame:
    """Query BigQuery and return a Pandas DataFrame.

    Parameters
    ----------
    table_name : str
        The name of the table to query.
    historical_days : int
        The number of days to query.

    Returns
    -------
    pd.DataFrame
        The query results.
    """

    # Check Cache
    df = cached(table_name)
    if df is not None:
        return df

    service_account_path = None

    # Check that service account file exists, check in current and parent directories.  If found, set path.
    if os.path.exists("service_account.json"):
        service_account_path = "service_account.json"
    elif os.path.exists("../service_account.json"):
        service_account_path = "../service_account.json"
    else:
        raise ValueError(
            "Service account file not found. Please ensure that service_account.json is in the current or parent directory."
        )

    # Load your service account credentials
    credentials = service_account.Credentials.from_service_account_file(
        service_account_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )

    project_id = table_name.split(".")[0]

    # Initialize a BigQuery client
    client = bigquery.Client(credentials=credentials, project=project_id)

    query = QUERY_TEMPLATE.format(
        table_name=table_name, historical_days=historical_days
    )

    # Run the query
    query_job = client.query(query)

    # Wait for the query to finish
    results = query_job.result()

    # Convert the results to a Pandas DataFrame
    df = cache(results.to_dataframe(), table_name)

    return df


def cache(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """Cache a Pandas DataFrame to a CSV file."""
    fn = re.sub(r"\W", "_", table_name) + ".csv"
    file_path = os.path.join("cache", fn)
    df.to_csv(file_path, index=False)
    return df


def cached(table_name: str) -> pd.DataFrame:
    """Check if a cached CSV file exists, and return a Pandas DataFrame if it does."""
    fn = re.sub(r"\W", "_", table_name) + ".csv"
    file_path = os.path.join("cache", fn)

    # Check for cache folder, create if it doesn't exist
    os.makedirs("cache", exist_ok=True)

    # Check for cached file, return if it exists
    if os.path.exists(file_path):
        return pd.read_csv(file_path, low_memory=False)

    return None


# Path: db.py
