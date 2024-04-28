"""


"""

import pandas as pd

# Create a sample DataFrame
data = {
    'Date': ['2022-01-01', '2022-01-01', '2022-01-02', '2022-01-02'],
    'City': ['New York', 'Los Angeles', 'New York', 'Los Angeles'],
    'Temperature': [32, 75, 30, 72],
    'Humidity': [60, 40, 55, 45]
}
df = pd.DataFrame(data)

# Display the original DataFrame
print("Original DataFrame:")
print(df)

# Reshape the DataFrame using pivot_table() to get average temperature and humidity by city
pivot_df = df.pivot_table(index='Date', columns='City', values=['Temperature', 'Humidity'], aggfunc='mean')

# Display the pivoted DataFrame
print("\nPivoted DataFrame:")
print(pivot_df)

# Reshape the DataFrame using melt() to convert columns into rows
melted_df = pd.melt(df, id_vars=['Date', 'City'], var_name='Metric', value_name='Value')

# Display the melted DataFrame
print("\nMelted DataFrame:")
print(melted_df)

# Reshape the DataFrame using stack() to pivot the innermost level of the column labels
stacked_df = pivot_df.stack()

# Display the stacked DataFrame
print("\nStacked DataFrame:")
print(stacked_df)

# Reshape the DataFrame using unstack() to pivot the innermost level of the index labels
unstacked_df = stacked_df.unstack()

# Display the unstacked DataFrame
print("\nUnstacked DataFrame:")
print(unstacked_df)
