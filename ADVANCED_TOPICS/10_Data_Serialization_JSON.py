# FileName: 10_Data_Serialization_JSON.py
# Code Written by Manuel Cortés Granados
# Date: 2024-03-04
# Location: Bogota DC, Colombia
# LinkedIn Profile: https://www.linkedin.com/in/mcortesgranados/

# This Python code demonstrates data serialization using JSON. It includes a dictionary data to be serialized, 
# and utilizes the json.dumps() method to serialize the dictionary to a JSON string. 
# It also utilizes the json.loads() method to deserialize the JSON string back to a dictionary. 
# Additionally, it includes metadata such as the proposed filename, authorship information, date and time, 
# location, and a link to the author's LinkedIn profile for context.

# Data Serialization using JSON in Python

# JSON (JavaScript Object Notation) is a lightweight data interchange format commonly used for data serialization and transmission in web applications.

import json

# Example dictionary to be serialized
data = {
    "name": "John",
    "age": 30,
    "city": "New York"
}

# Serialize the dictionary to JSON string
json_string = json.dumps(data)

# Output the serialized JSON string
print("Serialized JSON string:", json_string)

# Deserialize the JSON string to a dictionary
parsed_data = json.loads(json_string)

# Output the deserialized dictionary
print("Deserialized dictionary:", parsed_data)
