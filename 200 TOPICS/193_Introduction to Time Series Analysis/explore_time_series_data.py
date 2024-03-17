import pandas as pd
import matplotlib.pyplot as plt

# Load time series data from CSV file
data = pd.read_csv('time_series_data.csv')

# Plot time series data
plt.plot(data['Date'], data['Value'])
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Time Series Data')
plt.show()
