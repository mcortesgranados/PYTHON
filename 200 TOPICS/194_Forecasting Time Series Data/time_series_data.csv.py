import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_time_series_data(start_date, end_date, num_records):
    # Generate dates
    dates = pd.date_range(start=start_date, end=end_date, periods=num_records)

    # Generate random values for the time series
    values = [random.uniform(0, 100) for _ in range(num_records)]

    # Create a DataFrame with dates and values
    df = pd.DataFrame({'Date': dates, 'Value': values})

    return df

# Generate data
start_date = datetime(2022, 1, 1)
end_date = datetime(2022, 12, 31)
num_records = 1000000  # Adjust the number of records as needed
time_series_data = generate_time_series_data(start_date, end_date, num_records)

# Save data to CSV
time_series_data.to_csv('time_series_data.csv', index=False)
print("ok")