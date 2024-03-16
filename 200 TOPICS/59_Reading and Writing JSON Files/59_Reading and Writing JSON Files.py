
# Reading and writing JSON files in Python is a common task when working with data. JSON (JavaScript Object Notation) 
# is a lightweight data interchange format that is easy for humans to read and write, and easy for machines to parse and generate. 
# Python provides built-in libraries for working with JSON data. Here's how you can read from and write to JSON files in Python:

# Reading JSON from a File:

# Use the json.load() function to read JSON data from a file and parse it into a Python dictionary.
# Syntax: json.load(file_object)
# file_object: A file-like object (e.g., opened file or file-like object).

import json

def write_example_json_file():
    # Sample data
    data = {
        "name": "John Doe",
        "age": 30,
        "city": "New York",
        "languages": ["Python", "JavaScript", "Java"],
        "is_student": False,
        "grades": {"math": 90, "science": 85, "history": 88}
    }

    # Write the data to a JSON file
    with open('data.json', 'w') as file:
        json.dump(data, file, indent=4)

    print("Example JSON file 'data.json' has been created.")

# Call the method to write the example JSON file
write_example_json_file()

import json

# Open the JSON file
with open('data.json', 'r') as file:
    data = json.load(file)

print(data)

# Now 'data' contains the parsed JSON data as a Python dictionary

# Writing JSON to a File:

# Use the json.dump() function to serialize a Python dictionary into JSON format and write it to a file.
# Syntax: json.dump(data, file_object)
# data: The Python dictionary to be serialized into JSON.
# file_object: A file-like object (e.g., opened file or file-like object).

import json

# Sample data
data = {'name': 'John', 'age': 30, 'city': 'New York'}

# Open the JSON file in write mode
with open('output.json', 'w') as file:
    json.dump(data, file)

# Pretty-printing JSON:

# For better readability, you can use the indent parameter to add indentation and sort_keys parameter to sort keys alphabetically while writing JSON.
# Syntax: json.dump(data, file_object, indent=4, sort_keys=True)
# Example:

import json

# Sample data
data = {'name': 'John', 'age': 30, 'city': 'New York'}

# Open the JSON file in write mode
with open('output.json', 'w') as file:
    json.dump(data, file, indent=4, sort_keys=True)




