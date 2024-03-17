import pandas as pd

# Read data from CSV file
df = pd.read_csv('data.csv')

# Filter data based on conditions
filtered_data = df[df['Column1'] > 10]

# Display filtered data
print(filtered_data.head())
