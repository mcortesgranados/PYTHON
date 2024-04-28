import pandas as pd

# Create a sample DataFrame with hierarchical index
data = {
    'Value': [10, 20, 30, 40, 50, 60],
    'Region': ['North', 'North', 'South', 'South', 'East', 'East'],
    'Year': [2019, 2020, 2019, 2020, 2019, 2020],
}
df = pd.DataFrame(data)

# Set multiple columns as index to create hierarchical indexing
df.set_index(['Region', 'Year'], inplace=True)

# Display the original DataFrame with hierarchical index
print("Original DataFrame with Multi-indexing:")
print(df)

# Accessing data using multi-indexing
print("\nAccessing data using multi-indexing:")
print("Value for North region, Year 2020:", df.loc[('North', 2020)])

# Perform groupby operation on multi-indexed DataFrame
grouped = df.groupby(level='Region').sum()

# Display the grouped DataFrame
print("\nGrouped DataFrame:")
print(grouped)

# Resetting the index of the DataFrame
df_reset = df.reset_index()

# Display the DataFrame after resetting the index
print("\nDataFrame after resetting the index:")
print(df_reset)
