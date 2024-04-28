import pandas as pd

# Create a sample DataFrame
data = {
    'City': ['New York', 'Los Angeles', 'Chicago', 'New York', 'Los Angeles', 'Chicago'],
    'Month': ['Jan', 'Jan', 'Jan', 'Feb', 'Feb', 'Feb'],
    'Sales': [1000, 1500, 1200, 1300, 1600, 1100]
}
df = pd.DataFrame(data)

# Display the original DataFrame
print("Original DataFrame:")
print(df)

# Group the DataFrame by 'City' and calculate total sales for each city
grouped_df = df.groupby('City').agg(total_sales=('Sales', 'sum'))

# Display the grouped and aggregated DataFrame
print("\nGrouped and Aggregated DataFrame:")
print(grouped_df)

# Group the DataFrame by 'City' and 'Month' and calculate average sales for each combination
grouped_month_df = df.groupby(['City', 'Month']).agg(average_sales=('Sales', 'mean'))

# Display the grouped and aggregated DataFrame
print("\nGrouped and Aggregated DataFrame by City and Month:")
print(grouped_month_df)
