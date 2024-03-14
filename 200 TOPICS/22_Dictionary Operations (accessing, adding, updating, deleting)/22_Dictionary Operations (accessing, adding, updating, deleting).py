# 22_Dictionary Operations (accessing, adding, updating, deleting)
# Dictionaries in Python are a built-in data type used to store collections of key-value pairs. Below is a Python code sample demonstrating the creation and usage of dictionaries:

# In this code:

# Values are accessed using keys with square brackets [].
# New key-value pairs are added using assignment.
# Existing values are updated by reassigning them.
# Key-value pairs are deleted using the del statement or the pop() method.
# The pop() method removes and returns the value associated with the specified key.
# Iteration over key-value pairs is achieved using a for loop and the items() method.
# These operations provide flexibility in manipulating dictionaries and are commonly used for storing and accessing data in Python programs.

# Creating a dictionary
my_dict = {'name': 'John', 'age': 30, 'city': 'New York'}
print("Original dictionary:", my_dict)

# Accessing values
print("\nAccessing values:")
print("Name:", my_dict['name'])
print("Age:", my_dict['age'])

# Adding a new key-value pair
my_dict['gender'] = 'Male'
print("\nDictionary after adding 'gender':", my_dict)

# Updating a value
my_dict['age'] = 35
print("\nDictionary after updating 'age':", my_dict)

# Deleting a key-value pair
del my_dict['city']
print("\nDictionary after deleting 'city':", my_dict)

# Using pop() method to remove and return a value
removed_value = my_dict.pop('gender')
print("\nRemoved value:", removed_value)
print("Dictionary after pop('gender'):", my_dict)

# Iterating over key-value pairs
print("\nIterating over key-value pairs:")
for key, value in my_dict.items():
    print(key, ":", value)
