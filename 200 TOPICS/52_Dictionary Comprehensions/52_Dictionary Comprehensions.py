# 
# Dictionary comprehensions in Python provide a concise way to create dictionaries from iterable objects (such as lists, tuples, or ranges) using
#  a compact syntax. They allow you to generate key-value pairs by applying an expression to each item in the iterable and filtering the
#  items based on a condition. Dictionary comprehensions offer a more readable and compact alternative to traditional loops and conditional 
# statements for creating dictionaries.

# Here's the general syntax of a dictionary comprehension:

# new_dict = {key_expression: value_expression for item in iterable if condition}

# key_expression: The expression that determines the key of each key-value pair in the new dictionary.
# value_expression: The expression that determines the value of each key-value pair in the new dictionary.
# item: The variable representing each item in the iterable.
# iterable: The iterable (e.g., list, tuple, range) from which elements are drawn.
# condition (optional): An optional condition that filters the items based on some criteria. Only items for which the condition evaluates to True are included in the new dictionary.
# Example 1: Creating a dictionary of squares using a dictionary comprehension:

numbers = [1, 2, 3, 4, 5]
squared_dict = {x: x**2 for x in numbers}
print(squared_dict)  # Output: {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}


# Example 2: Filtering even numbers using a dictionary comprehension:


numbers = [1, 2, 3, 4, 5]
even_dict = {x: x for x in numbers if x % 2 == 0}
print(even_dict)  # Output: {2: 2, 4: 4}


# Example 3: Creating a dictionary from a list of tuples using a dictionary comprehension:



pairs = [('a', 1), ('b', 2), ('c', 3)]
dict_from_tuples = {key: value for key, value in pairs}
print(dict_from_tuples)  # Output: {'a': 1, 'b': 2, 'c': 3}


# Dictionary comprehensions can also be nested and include multiple for loops and conditional expressions for more complex transformations. 
# As with list comprehensions, it's essential to maintain readability and avoid excessive complexity to ensure that the code remains understandable.





