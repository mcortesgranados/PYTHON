"""
01. Loading and reading data from various file formats like CSV, Excel, SQL, JSON, etc.

"""

import pandas as pd

# Load data from a CSV file
csv_data = pd.read_csv('data.csv')

# Load data from an Excel file
excel_data = pd.read_excel('data.xlsx')

# Load data from a SQL database using SQLAlchemy
from sqlalchemy import create_engine
engine = create_engine('sqlite:///data.db')
sql_data = pd.read_sql('SELECT * FROM table_name', con=engine)

# Load data from a JSON file
json_data = pd.read_json('data.json')

# Display the loaded data
print("CSV Data:")
print(csv_data.head())

print("\nExcel Data:")
print(excel_data.head())

print("\nSQL Data:")
print(sql_data.head())

print("\nJSON Data:")
print(json_data.head())
