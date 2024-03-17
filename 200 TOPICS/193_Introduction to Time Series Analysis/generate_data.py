import pandas as pd
import numpy as np

# Generate sample date range
date_range = pd.date_range(start='2022-01-01', periods=1000000, freq='T')

# Generate sample values
values = np.random.normal(loc=100, scale=20, size=len(date_range))

# Create DataFrame
data = pd.DataFrame({'Date': date_range, 'Value': values})

# Save DataFrame to CSV file
data.to_csv('time_series_data.csv', index=False)

print("Data generated and saved successfully.")
