"""
03. Selecting and indexing data based on labels or positions using .loc[], .iloc[], etc.
"""

import pandas as pd

# Sample data: Create a DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age': [24, 27, 22, 32, 29],
    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'],
    'Salary': [50000, 60000, 55000, 70000, 65000]
}

# Create DataFrame
df = pd.DataFrame(data)

# 1. Selecting data by row label using .loc[]
# Select the row where the label is 2 (Charlie)
print("Selecting row with label 2 using .loc[]:")
print(df.loc[2])

# 2. Selecting data by multiple row labels using .loc[]
# Select rows 1, 3, and 4
print("\nSelecting rows with labels 1, 3, and 4 using .loc[]:")
print(df.loc[[1, 3, 4]])

# 3. Selecting specific columns by labels using .loc[]
# Select the 'Name' and 'Age' columns for the first three rows
print("\nSelecting 'Name' and 'Age' columns for the first three rows using .loc[]:")
print(df.loc[0:2, ['Name', 'Age']])

# 4. Selecting data by integer position using .iloc[]
# Select the row at position 2 (Charlie)
print("\nSelecting row at position 2 using .iloc[]:")
print(df.iloc[2])

# 5. Selecting data by multiple integer positions using .iloc[]
# Select rows at positions 0, 2, and 4
print("\nSelecting rows at positions 0, 2, and 4 using .iloc[]:")
print(df.iloc[[0, 2, 4]])

# 6. Selecting specific columns by position using .iloc[]
# Select columns at positions 1 and 3 for the first three rows
print("\nSelecting columns at positions 1 and 3 for the first three rows using .iloc[]:")
print(df.iloc[0:3, [1, 3]])

# 7. Using .iloc[] with both row and column positions
# Select the row at position 1 (Bob) and column at position 2 (City)
print("\nSelecting row at position 1 and column at position 2 using .iloc[]:")
print(df.iloc[1, 2])
