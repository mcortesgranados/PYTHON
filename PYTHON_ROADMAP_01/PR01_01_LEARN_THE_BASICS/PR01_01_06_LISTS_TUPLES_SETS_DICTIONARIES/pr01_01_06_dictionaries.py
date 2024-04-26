# This program demonstrates working with dictionaries in Python

# Dictionaries (unordered key-value pairs)

# Dictionaries are collections of key-value pairs, similar to real-world dictionaries. 
# Each key acts as a unique identifier for its associated value. Keys must be immutable 
# (like strings or numbers), while values can be of any data type.

# Defining a dictionary
person_info = {
  "name": "Alice",
  "age": 30,
  "city": "New York",
  "occupation": "Software Engineer"  # You can add more key-value pairs
}

# Accessing elements by key
name = person_info["name"]
age = person_info["age"]

print("Name:", name)
print("Age:", age)

# Checking for key existence (avoiding errors)
if "occupation" in person_info:
  print("Occupation:", person_info["occupation"])
else:
  print("Occupation key not found.")

# Modifying values using key
person_info["city"] = "Seattle"  # Change city to "Seattle"

print("Updated dictionary (city changed):", person_info)

# Adding new key-value pair
person_info["hobbies"] = ["reading", "hiking"]  # Add hobbies as a list

print("Dictionary with hobbies:", person_info)

# Removing key-value pair
del person_info["age"]  # Remove the "age" key-value pair

print("Dictionary without age:", person_info)

# Looping through key-value pairs
for key, value in person_info.items():
  print(f"{key}: {value}")  # f-string for formatted printing

# Getting all keys or values as separate lists
keys = list(person_info.keys())
values = list(person_info.values())

print("List of keys:", keys)
print("List of values:", values)
