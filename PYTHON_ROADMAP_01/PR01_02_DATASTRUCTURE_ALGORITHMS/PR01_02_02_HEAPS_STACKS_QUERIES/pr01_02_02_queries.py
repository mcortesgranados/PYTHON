# Python Queries Example

# Example 1: Filtering Elements
# Filtering involves selecting elements from a collection that satisfy a given condition.
# Python provides several methods for filtering, such as list comprehensions, filter(), and lambda functions.
# Here's an example of filtering even numbers from a list using list comprehensions:

numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
even_numbers = [x for x in numbers if x % 2 == 0]
# Explanation: This uses a list comprehension to create a new list containing only even numbers.

print("Even Numbers:", even_numbers)

# Example 2: Searching Elements
# Searching involves finding elements in a collection that match a specific value or condition.
# Python provides methods like index() and find() for searching elements in lists and strings, respectively.
# Here's an example of searching for the index of the first occurrence of a value in a list:

index = numbers.index(5)
# Explanation: This finds the index of the first occurrence of the value 5 in the 'numbers' list.

print("Index of 5:", index)

# Example 3: Transforming Elements
# Transformation involves applying a function to each element in a collection to produce a new collection.
# Python provides methods like map() and list comprehensions for transforming elements.
# Here's an example of squaring each element in a list using map() and a lambda function:

squared_numbers = list(map(lambda x: x**2, numbers))
# Explanation: This uses the map() function with a lambda function to square each element in the 'numbers' list.

print("Squared Numbers:", squared_numbers)

# Example 4: Combining Filtering and Transformation
# You can combine filtering and transformation to perform more complex queries.
# Here's an example of filtering even numbers and then squaring them using a list comprehension:

even_squared_numbers = [x**2 for x in numbers if x % 2 == 0]
# Explanation: This uses a list comprehension to filter even numbers and square them simultaneously.

print("Even Squared Numbers:", even_squared_numbers)

# Documenting the Queries:
def queries_documentation():
    """
    This function demonstrates various query operations in Python.

    Example 1:
    - Filtering Elements: How to filter elements from a collection based on a condition.

    Example 2:
    - Searching Elements: How to search for elements in a collection.

    Example 3:
    - Transforming Elements: How to transform elements of a collection using a function.

    Example 4:
    - Combining Filtering and Transformation: How to combine filtering and transformation operations.
    """
    pass

# End of example
