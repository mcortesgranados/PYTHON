import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Function to generate random names
def generate_name():
    names = ['Alice', 'Bob', 'Charlie', 'David', 'Eva', 'Frank', 'Grace', 'Henry', 'Ivy', 'Jack']
    surnames = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez']
    return random.choice(names), random.choice(surnames)

# Function to generate random gender orientations
def generate_orientation():
    orientations = ['Straight', 'Gay', 'Lesbian', 'Bisexual', 'Pansexual']
    return random.choice(orientations)

# Generate sample data
data = {
    'Column1': np.random.randint(1, 100, size=10000),
    'Column2': np.random.randint(50, 200, size=10000),
    'Category': np.random.choice(['A', 'B', 'C'], size=10000),
    'Value': np.random.randint(100, 500, size=10000),
    'Month': np.random.choice(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], size=10000),
    'Date': [datetime.today() - timedelta(days=random.randint(1, 365*5)) for _ in range(10000)]
}

# Generate random names, surnames, and gender orientations for new columns
names_and_surnames = [generate_name() for _ in range(10000)]
data['Name'] = [name for name, _ in names_and_surnames]
data['Surname'] = [surname for _, surname in names_and_surnames]
data['Orientation'] = [generate_orientation() for _ in range(10000)]

# Create DataFrame
df = pd.DataFrame(data)

# Write DataFrame to CSV file
df.to_csv('data.csv', index=False)

print("Data CSV file with 10,000 records generated successfully.")
