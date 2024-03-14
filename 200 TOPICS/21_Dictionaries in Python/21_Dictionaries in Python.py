# Dictionaries in Python are a built-in data type used to store collections of key-value pairs. 
# Below is a Python code sample demonstrating the creation and usage of dictionaries:

# In this code:

# Dictionaries are created using curly braces {} and key-value pairs separated by colons :.
# Values are accessed using the keys.
# New key-value pairs can be added using assignment.
# Existing values can be modified by reassigning them.
# Key-value pairs can be removed using the pop() method.
# Dictionaries support iteration over their key-value pairs using a for loop and the items() method.
# Dictionaries are versatile data structures that allow you to store and retrieve data efficiently using unique keys. They are commonly used for tasks like storing settings, mapping IDs to values, and more.

# Creating a dictionary
my_dict = {'name': 'John', 'age': 30, 'city': 'New York'}
print("Dictionary:", my_dict)

# Accessing values
print("Value associated with 'name':", my_dict['name'])
print("Value associated with 'age':", my_dict['age'])

# Adding a new key-value pair
my_dict['gender'] = 'Male'
print("Dictionary after adding 'gender':", my_dict)

# Modifying a value
my_dict['age'] = 35
print("Dictionary after modifying 'age':", my_dict)

# Removing a key-value pair
removed_value = my_dict.pop('city')
print("Removed value:", removed_value)
print("Dictionary after removing 'city':", my_dict)

# Iterating over key-value pairs
print("Iterating over key-value pairs:")
for key, value in my_dict.items():
    print(key, ":", value)
