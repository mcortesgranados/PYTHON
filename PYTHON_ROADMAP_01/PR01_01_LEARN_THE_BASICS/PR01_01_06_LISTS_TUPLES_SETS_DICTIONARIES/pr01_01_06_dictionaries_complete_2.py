# Python Dictionaries Example

# Example 1: Creating a Dictionary
# Dictionaries in Python are unordered collections of key-value pairs.
# They are defined using curly braces {} and can contain any immutable data type as keys and any data type as values.
# Here's an example of creating a dictionary:
person = {
    "name": "John",
    "age": 30,
    "city": "New York"
}

# Example 2: Accessing Values in a Dictionary
# You can access the values in a dictionary using the keys.
# Here's how you can access the name and age of the person:
name = person["name"]
age = person["age"]
print("Name:", name)
print("Age:", age)

# Example 3: Adding and Updating Values in a Dictionary
# You can add new key-value pairs to a dictionary or update existing ones.
# Here's how you can add/update the city of the person:
person["city"] = "Los Angeles"  # Updating the city
person["gender"] = "Male"       # Adding a new key-value pair
print("Updated Dictionary:", person)

# Example 4: Removing Items from a Dictionary
# You can remove items from a dictionary using the del keyword or the pop() method.
# Here's how you can remove the age of the person using del and pop():
del person["age"]        # Using del keyword
removed_city = person.pop("city")  # Using pop() method
print("Dictionary after removal:", person)
print("Removed City:", removed_city)

# Example 5: Iterating Over a Dictionary
# You can iterate over the key-value pairs of a dictionary using a for loop.
# Here's how you can iterate over the key-value pairs of the person dictionary:
for key, value in person.items():
    print(key + ":", value)

# Example 6: Dictionary Methods
# Python provides several methods to manipulate dictionaries.
# Here are some commonly used dictionary methods:
# - keys(): Returns a view of all keys in the dictionary.
# - values(): Returns a view of all values in the dictionary.
# - items(): Returns a view of all key-value pairs in the dictionary as tuples.
# - clear(): Removes all items from the dictionary.
# - get(): Returns the value associated with a specified key. If the key does not exist, it returns None or a default value.
# - update(): Updates the dictionary with key-value pairs from another dictionary or iterable.
# - copy(): Returns a shallow copy of the dictionary.
keys = person.keys()
values = person.values()
items = person.items()
print("Keys:", keys)
print("Values:", values)
print("Items:", items)

# Example 7: Nested Dictionaries
# You can have dictionaries within a dictionary, known as nested dictionaries.
# Here's an example of a nested dictionary:
employees = {
    "John": {"age": 30, "position": "Manager"},
    "Alice": {"age": 25, "position": "Developer"}
}

# Example 8: Dictionary Comprehension
# Similar to list comprehension, you can also use dictionary comprehension to create dictionaries in a concise way.
# Here's an example of dictionary comprehension to create a dictionary of squares:
squares = {x: x**2 for x in range(5)}
print("Squares:", squares)

# Documenting the Dictionary:
def dictionary_documentation():
    """
    This function demonstrates various aspects of dictionaries in Python.

    Example 1:
    - Creating a Dictionary: How to define a dictionary in Python.

    Example 2:
    - Accessing Values in a Dictionary: How to access values using keys.

    Example 3:
    - Adding and Updating Values in a Dictionary: How to add new key-value pairs or update existing ones.

    Example 4:
    - Removing Items from a Dictionary: How to remove items using del keyword or pop() method.

    Example 5:
    - Iterating Over a Dictionary: How to iterate over key-value pairs using a for loop.

    Example 6:
    - Dictionary Methods: Commonly used methods to manipulate dictionaries.

    Example 7:
    - Nested Dictionaries: How to create dictionaries within dictionaries.

    Example 8:
    - Dictionary Comprehension: How to create dictionaries using dictionary comprehension.
    """
    pass

# End of examples
