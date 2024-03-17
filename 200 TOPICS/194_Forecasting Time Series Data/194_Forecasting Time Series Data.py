

import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt

# Load the time series data
data = pd.read_csv('time_series_data.csv')

# Preprocess the data (Prophet expects columns named 'ds' for the time series and 'y' for the values)
data['ds'] = pd.to_datetime(data['Date'])
data.rename(columns={'Value': 'y'}, inplace=True)

# Instantiate Prophet model
model = Prophet()

# Fit the model
model.fit(data)

# Make future predictions
future = model.make_future_dataframe(periods=365)  # Forecast for 1 year (365 days) into the future
forecast = model.predict(future)

# Plot the forecast
fig = model.plot(forecast)
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Forecasted Time Series Data')
plt.show()
