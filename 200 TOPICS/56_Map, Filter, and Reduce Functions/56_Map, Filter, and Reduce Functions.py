#  The map(), filter(), and reduce() functions are built-in functions in Python for performing common operations on iterables like lists, 
# tuples, or sets. These functions are often used in combination with lambda functions to apply simple operations to each element of an iterable or
#  to filter elements based on certain conditions. Here's a brief overview of each:

# map(function, iterable): The map() function applies a given function to each item of an iterable (like a list) and returns an iterator 
# that yields the results. It applies the function to each item in the iterable and returns a new iterator with the results.

# map(function, iterable)

# function: The function to apply to each item in the iterable.
# iterable: The iterable (e.g., list, tuple) whose items will be passed to the function.

numbers = [1, 2, 3, 4, 5]
squared_numbers = map(lambda x: x**2, numbers)
print(list(squared_numbers))  # Output: [1, 4, 9, 16, 25]


# filter(function, iterable): The filter() function creates an iterator that filters elements from an iterable based on a given function (predicate) 
# that returns True or False. It returns an iterator containing only the elements for which the function returns True.

# Syntax: filter(function, iterable)
# function: The function (predicate) used to filter elements.
# iterable: The iterable (e.g., list, tuple) to be filtered.
# Example:
numbers = [1, 2, 3, 4, 5]
even_numbers = filter(lambda x: x % 2 == 0, numbers)
print(list(even_numbers))  # Output: [2, 4]


# reduce(function, iterable[, initializer]): The reduce() function applies a rolling computation to sequential pairs of values in an iterable. 
# It repeatedly applies the function to the elements of the iterable, reducing them to a single value. The initializer argument is optional and provides an 
# initial value for the computation.

# Note: In Python 3, reduce() is no longer a built-in function and is available in the functools module.

# Syntax:

# reduce(function, iterable[, initializer])
# function: The function used for the rolling computation.
# iterable: The iterable (e.g., list, tuple) whose elements are reduced.
# initializer (optional): An initial value for the computation.
# Example:
from functools import reduce
numbers = [1, 2, 3, 4, 5]
product = reduce(lambda x, y: x * y, numbers)
print(product)  # Output: 120 (1 * 2 * 3 * 4 * 5)


# These functions are powerful tools for functional programming in Python and can be used to express complex operations in a concise and readable manner.
#  When combined with lambda functions, they offer a flexible way to manipulate and process data. However, it's essential to use them 
# judiciously and consider readability when applying them in code.