# Python Lambdas Example

# Example 1: Creating a Simple Lambda Function
# Lambda functions are defined using the lambda keyword and can have any number of arguments.
# They are often used for small, single-expression functions.
# Here's an example of creating a simple lambda function that adds two numbers:

add = lambda x, y: x + y
# Explanation: This defines a lambda function that takes two arguments (x and y) and returns their sum.
# The syntax is lambda arguments: expression.

result = add(3, 5)
print("Result (Addition):", result)

# Example 2: Using Lambdas with Built-in Functions
# Lambdas are commonly used with built-in functions like map(), filter(), and sorted().
# They provide a concise way to specify the transformation or condition.
# Here's an example of using a lambda with map() to square each element in a list:

numbers = [1, 2, 3, 4, 5]
squared_numbers = list(map(lambda x: x**2, numbers))
# Explanation: This uses a lambda function to square each element in the numbers list.
# The map() function applies the lambda function to each element and returns an iterator,
# which is converted to a list using list().

print("Squared Numbers:", squared_numbers)

# Example 3: Sorting with Lambdas
# Lambdas are also useful for defining custom sorting keys when using the sorted() function.
# Here's an example of using a lambda to sort a list of tuples based on the second element:

points = [(1, 2), (3, 1), (2, 3)]
sorted_points = sorted(points, key=lambda x: x[1])
# Explanation: This uses a lambda function as the key argument to sorted().
# The lambda function extracts the second element of each tuple for sorting.

print("Sorted Points (By Y-coordinate):", sorted_points)

# Example 4: Using Lambdas with Conditional Expressions
# Lambdas can also include conditional expressions for more complex behavior.
# Here's an example of using a lambda with a conditional expression to categorize numbers:

categorize = lambda x: 'even' if x % 2 == 0 else 'odd'
# Explanation: This lambda function categorizes numbers as 'even' or 'odd' based on the condition.
# It uses a conditional expression 'if x % 2 == 0 else' to determine the category.

print("Category of 6:", categorize(6))
print("Category of 7:", categorize(7))

# Documenting the Lambdas:
def lambdas_documentation():
    """
    This function demonstrates various aspects of lambdas in Python.

    Example 1:
    - Creating a Simple Lambda Function: How to create a lambda function for adding two numbers.

    Example 2:
    - Using Lambdas with Built-in Functions: How to use lambdas with map() to transform elements in a list.

    Example 3:
    - Sorting with Lambdas: How to use lambdas to define custom sorting keys for sorted().

    Example 4:
    - Using Lambdas with Conditional Expressions: How to use lambdas with conditional expressions for categorizing numbers.
    """
    pass

# End of examples
