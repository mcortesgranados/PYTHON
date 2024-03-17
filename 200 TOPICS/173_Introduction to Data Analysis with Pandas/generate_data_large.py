# generate_data_large.py

import csv
import random

# Define the number of rows per chunk
chunk_size = 10000
total_rows = 100000000

# Open the CSV file in append mode
with open('data_large.csv', 'a', newline='') as csvfile:
    fieldnames = ['Column1', 'Column2', 'Category', 'Value', 'Month']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # Write the header if the file is empty
    if csvfile.tell() == 0:
        writer.writeheader()

    # Generate and write data in chunks
    for _ in range(total_rows // chunk_size):
        data = []
        for _ in range(chunk_size):
            row = {
                'Column1': random.randint(1, 100),
                'Column2': random.randint(50, 200),
                'Category': random.choice(['A', 'B', 'C']),
                'Value': random.randint(100, 500),
                'Month': random.choice(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            }
            data.append(row)

        writer.writerows(data)

print("Data CSV file generated successfully.")
