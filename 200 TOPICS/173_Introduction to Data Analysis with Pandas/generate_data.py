# generate_data.py

import csv
import random

# Generate sample data
data = []
for i in range(100000000):
    row = {
        'Column1': random.randint(1, 100),
        'Column2': random.randint(50, 200),
        'Category': random.choice(['A', 'B', 'C']),
        'Value': random.randint(100, 500),
        'Month': random.choice(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    }
    data.append(row)

# Write data to CSV file
with open('data.csv', 'w', newline='') as csvfile:
    fieldnames = ['Column1', 'Column2', 'Category', 'Value', 'Month']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for row in data:
        writer.writerow(row)

print("Data CSV file generated successfully.")
