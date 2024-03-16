# 
# Set comprehensions in Python provide a concise way to create sets from iterable objects (such as lists, tuples, or ranges) 
# using a compact syntax. They allow you to generate set elements by applying an expression to each item in the iterable and 
# filtering the items based on a condition. Set comprehensions offer a more readable and compact alternative to traditional loops and 
# conditional statements for creating sets.

# Here's the general syntax of a set comprehension:

# new_set = {expression for item in iterable if condition}

# expression: The expression that determines the value of each element in the new set.
# item: The variable representing each item in the iterable.
# iterable: The iterable (e.g., list, tuple, range) from which elements are drawn.
# condition (optional): An optional condition that filters the items based on some criteria. Only items for which the condition evaluates to True are 
# included in the new set.
# Example 1: Creating a set of squares using a set comprehension:

numbers = {1, 2, 3, 4, 5}
squared_set = {x**2 for x in numbers}
print(squared_set)  # Output: {1, 4, 9, 16, 25}


# Example 2: Filtering even numbers using a set comprehension:

numbers = {1, 2, 3, 4, 5}
even_set = {x for x in numbers if x % 2 == 0}
print(even_set)  # Output: {2, 4}


# Example 3: Creating a set from a list of tuples using a set comprehension:

pairs = [('a', 1), ('b', 2), ('c', 3)]
set_from_tuples = {key for key, value in pairs}
print(set_from_tuples)  # Output: {'a', 'b', 'c'}


# Set comprehensions can also be nested and include multiple for loops and conditional expressions for more complex transformations. 
# As with list and dictionary comprehensions, it's essential to maintain readability and avoid excessive complexity to ensure that the 
# code remains understandable.







