"""
04. Filtering and querying data based on conditions using boolean indexing or query methods.

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

# 1. Filtering using boolean indexing
# Filter for rows where Age is greater than 25
print("Rows where Age > 25 using boolean indexing:")
print(df[df['Age'] > 25])

# 2. Filtering using multiple conditions with boolean indexing
# Filter for rows where Age is greater than 25 and Salary is greater than 60000
print("\nRows where Age > 25 and Salary > 60000 using boolean indexing:")
print(df[(df['Age'] > 25) & (df['Salary'] > 60000)])

# 3. Filtering using the query() method
# Query for rows where the City is 'Chicago' or 'Phoenix'
print("\nRows where City is 'Chicago' or 'Phoenix' using query():")
print(df.query("City == 'Chicago' or City == 'Phoenix'"))

# 4. Filtering with negation using boolean indexing
# Filter for rows where Age is not equal to 27
print("\nRows where Age is not equal to 27 using negation in boolean indexing:")
print(df[df['Age'] != 27])

# 5. Filtering using the query() method with more complex conditions
# Query for rows where Salary is greater than 55000 and Age is less than 30
print("\nRows where Salary > 55000 and Age < 30 using query():")
print(df.query("Salary > 55000 and Age < 30"))
