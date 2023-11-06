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
        parse_date('%Y%m%d', event_date) event_date,
        event_name,
        timestamp_micros(event_timestamp) as event_timestamp,
        user_pseudo_id,
        timestamp_micros(user_first_touch_timestamp) as user_first_touch_timestamp,
        geo.region as geo_region,
        geo.city as geo_city,
        traffic_source.name as channel,
        traffic_source.source as source,
        max(if(params.key = 'ga_session_id', params.value.int_value, null)) ga_session_id,
        max(if(params.key = 'ga_session_number', params.value.int_value, null)) ga_session_number,
        cast(max(if(params.key = 'session_engaged', params.value.string_value, null)) as int64) session_engaged,
        max(if(params.key = 'page_location', params.value.string_value, null)) page_location,
        max(if(params.key = 'page_title', params.value.string_value, null)) page_title,
        FROM raw_ga_4,
        UNNEST(event_params) AS params
        GROUP BY event_date, event_name, event_timestamp, user_pseudo_id, user_first_touch_timestamp, channel, source, geo_region, geo_city
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

    # Load your service account credentials
    credentials = service_account.Credentials.from_service_account_file(
        "service_account.json",
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
        return pd.read_csv(file_path)

    return None


# Path: db.py
