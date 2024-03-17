import pandas as pd

# Create a DataFrame from a dictionary
data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35]}
df = pd.DataFrame(data)

# Access elements by column and row index
print(df['Name'])  # Access 'Name' column
print(df.loc[0])   # Access first row
