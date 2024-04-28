import pandas as pd
import numpy as np

# Create a sample time series DataFrame
date_range = pd.date_range(start='2022-01-01', end='2022-01-10')
data = {
    'Date': date_range,
    'Value': np.random.randint(0, 100, size=len(date_range))
}
df = pd.DataFrame(data)
df.set_index('Date', inplace=True)

# Display the original DataFrame
print("Original DataFrame:")
print(df)

# Resample the DataFrame to a weekly frequency, taking the sum of values
weekly_resampled = df.resample('W').sum()

# Display the resampled DataFrame
print("\nResampled DataFrame (Weekly):")
print(weekly_resampled)

# Compute the rolling mean with a window size of 3 days
rolling_mean = df.rolling(window=3).mean()

# Display the DataFrame with rolling mean
print("\nDataFrame with Rolling Mean:")
print(rolling_mean)

# Compute the exponential moving average with a span of 3 days
ema = df.ewm(span=3).mean()

# Display the DataFrame with exponential moving average
print("\nDataFrame with Exponential Moving Average:")
print(ema)
