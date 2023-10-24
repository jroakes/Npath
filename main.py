import pandas as pd
from data import get_data


table_name = "locomotive-ga4-data.analytics_334723581.events_*"
historical_days = 180
goal_event = "contact_us"

df = get_data(table_name, historical_days, goal_event)

print(df.describe().to_markdown())
print()
print(df.dtypes)
