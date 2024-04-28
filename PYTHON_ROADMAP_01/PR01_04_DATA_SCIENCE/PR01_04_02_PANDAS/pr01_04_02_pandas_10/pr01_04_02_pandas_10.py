import pandas as pd

# Create a sample DataFrame
data = {
    'A': [1, 2, 3, 4, 5],
    'B': [6, 7, 8, 9, 10],
    'C': [11, 12, 13, 14, 15]
}
df = pd.DataFrame(data)

# Display the original DataFrame
print("Original DataFrame:")
print(df)

# Define a custom function to double the value
def double_value(x):
    return x * 2

# Apply the custom function to each column using applymap()
result_applymap = df.applymap(double_value)

# Display the DataFrame after applying the custom function using applymap()
print("\nDataFrame after applying custom function using applymap():")
print(result_applymap)

# Define a custom function to calculate the square of the value
def square_value(x):
    return x ** 2

# Apply the custom function to each column using apply()
result_apply = df.apply(square_value)

# Display the DataFrame after applying the custom function using apply()
print("\nDataFrame after applying custom function using apply():")
print(result_apply)
