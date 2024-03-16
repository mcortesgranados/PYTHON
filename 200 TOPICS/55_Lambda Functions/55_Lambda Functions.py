# Lambda functions, also known as anonymous functions, are small, inline functions that are defined using a compact syntax in Python. 
# They are useful for creating simple, short-lived functions without the need for a formal def statement. 
# Lambda functions are often used in situations where a function is required as an argument to another function, such as in map(), 
# filter(), and sorted() functions, or in list comprehensions.

# Here's the general syntax of a lambda function:

# lambda arguments: expression


# arguments: The input parameters (arguments) of the lambda function.
# expression: The single expression that defines the operation of the lambda function. This expression is evaluated and returned as the result of the function.
# Lambda functions can have any number of arguments, but they can only contain a single expression. They are typically used for simple operations and are not 
# suitable for complex logic or multi-line functions.

# Example 1: Creating a lambda function to square a number:

square = lambda x: x**2
print(square(5))  # Output: 25


# Example 2: Using a lambda function as an argument to the map() function:



numbers = [1, 2, 3, 4, 5]
squared_numbers = map(lambda x: x**2, numbers)
print(list(squared_numbers))  # Output: [1, 4, 9, 16, 25]


# Example 3: Sorting a list of tuples based on the second element using a lambda function:


pairs = [('a', 3), ('b', 1), ('c', 2)]
sorted_pairs = sorted(pairs, key=lambda x: x[1])
print(sorted_pairs)  # Output: [('b', 1), ('c', 2), ('a', 3)]


# Lambda functions are particularly useful in situations where a small, one-off function is needed, such as in data manipulation or 
# transformation tasks. However, they should be used judiciously and sparingly, as they can make code less readable when overused or when used for complex logic.
