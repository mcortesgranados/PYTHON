import pandas as pd

# Read data from CSV file
df = pd.read_csv('data.csv')

# Convert 'Month' column to datetime dtype
df['Month'] = pd.to_datetime(df['Month'])

# Perform time series analysis (e.g., resampling, rolling mean)
# Example: Calculate monthly average of 'Value' column
monthly_avg = df.resample('M', on='Month').mean()

# Display the monthly average
print(monthly_avg)
