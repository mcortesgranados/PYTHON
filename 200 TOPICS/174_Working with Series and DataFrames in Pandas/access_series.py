import pandas as pd

# Create a Series from a list
data = [1, 2, 3, 4, 5]
s = pd.Series(data)

# Access elements by index
print(s[0])  # Access first element
print(s[2:4])  # Access elements from index 2 to 3
