"""
28. Plotting candlestick charts to visualize stock price data.

"""

import mplfinance as mpf
import pandas as pd

# Sample stock price data (OHLC format)
data = {
    'Date': ['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04', '2022-01-05'],
    'Open': [100, 102, 105, 98, 101],
    'High': [105, 110, 106, 104, 105],
    'Low': [98, 100, 95, 94, 99],
    'Close': [103, 108, 100, 99, 103],
    'Volume': [100000, 120000, 95000, 110000, 105000]
}

# Convert data to DataFrame
df = pd.DataFrame(data)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Plot candlestick chart
mpf.plot(df, type='candle', style='charles', volume=True)
