# Reading and writing CSV (Comma-Separated Values) files in Python is a common task for working with tabular data. CSV files are plain text 
# files that store tabular data with each line representing a row and each value separated by a comma (,). Python provides built-in libraries for reading and writing CSV files. Here's how you can read from and write to CSV files in Python:

# Reading CSV Files:

# Use the csv.reader object to read data from a CSV file.
# Syntax: csv.reader(file_object)
# file_object: A file-like object (e.g., opened file or file-like object).
# Example:

import csv

def write_example_csv_file():
    # Sample data
    data = [
        ['Name', 'Age', 'City'],
        ['John', 30, 'New York'],
        ['Alice', 25, 'Los Angeles'],
        ['Bob', 35, 'Chicago']
    ]

    # Write the data to a CSV file
    with open('data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

    print("Example CSV file 'data.csv' has been created.")

# Call the method to write the example CSV file
write_example_csv_file()

import csv

# Open the CSV file
with open('data.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        print(row)

# Reading and Writing CSV Files with Headers:

# You can use csv.DictReader and csv.DictWriter objects to handle CSV files with headers.
# csv.DictReader reads each row as a dictionary with keys derived from the CSV headers.
# csv.DictWriter writes data to a CSV file using dictionaries where keys correspond to column headers.
# Example:

import csv

# Reading CSV with headers
with open('data.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        print(row['Name'], row['Age'], row['City'])

# Writing CSV with headers
with open('output.csv', 'w', newline='') as file:
    fieldnames = ['Name', 'Age', 'City']
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({'Name': 'John', 'Age': 30, 'City': 'New York'})
    writer.writerow({'Name': 'Alice', 'Age': 25, 'City': 'Los Angeles'})
    writer.writerow({'Name': 'Bob', 'Age': 35, 'City': 'Chicago'})


