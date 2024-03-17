import pandas as pd

# Create a Series from a list
data = [1, 2, 3, 4, 5]
s = pd.Series(data)

# Calculate sum, mean, and max
print(s.sum())   # Sum of all elements
print(s.mean())  # Mean of all elements
print(s.max())   # Maximum value in the series
