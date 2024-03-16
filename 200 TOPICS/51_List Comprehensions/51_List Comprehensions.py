# 51_List Comprehensions


# List comprehensions are a concise way to create lists in Python. They allow you to generate a new list by applying an expression to each item in an existing
#  iterable (such as a list, tuple, or range) and filtering the items based on a condition. List comprehensions offer a more readable and compact
#  alternative to traditional loops and conditional statements.

# Here's the general syntax of a list comprehension:

# new_list = [expression for item in iterable if condition]


# expression: The expression to be evaluated for each item in the iterable. This expression determines the value of each element in the new list.
# item: The variable representing each item in the iterable.
# iterable: The iterable (e.g., list, tuple, range) from which elements are drawn.
# condition (optional): An optional condition that filters the items based on some criteria. Only items for which the condition evaluates to True are included in the new list.
# Example 1: Squaring numbers using a list comprehension:

numbers = [1, 2, 3, 4, 5]
squared = [x**2 for x in numbers]
print(squared)  # Output: [1, 4, 9, 16, 25]


# Example 2: Filtering even numbers using a list comprehension:


numbers = [1, 2, 3, 4, 5]
even_numbers = [x for x in numbers if x % 2 == 0]
print(even_numbers)  # Output: [2, 4]


# Example 3: Creating a list of tuples using a list comprehension:


names = ['Alice', 'Bob', 'Charlie']
name_lengths = [(name, len(name)) for name in names]
print(name_lengths)  # Output: [('Alice', 5), ('Bob', 3), ('Charlie', 7)]


# List comprehensions can be nested, and they can also include multiple for loops and conditional expressions for more complex transformations. However, 
# it's essential to maintain readability and avoid excessive complexity to ensure that the code remains understandable.