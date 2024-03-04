# FileName: 19_Data_Serialization_JSON.py
# Code Written by Manuel Cortés Granados
# Date: 2024-03-04
# Location: Bogota DC, Colombia
# LinkedIn Profile: https://www.linkedin.com/in/mcortesgranados/

# Data Serialization with JSON in Python

# JSON (JavaScript Object Notation) is a lightweight data-interchange format that is easy for humans to read and write.
# Python provides built-in support for JSON serialization and deserialization through the json module.

import json

# Sample dictionary data to be serialized
data = {
    'name': 'John',
    'age': 30,
    'city': 'New York'
}

# Serialize the dictionary to a JSON string
json_string = json.dumps(data)

print("Serialized JSON string:")
print(json_string)

# Deserialize the JSON string back to a dictionary
decoded_data = json.loads(json_string)

print("\nDeserialized data:")
print(decoded_data)
