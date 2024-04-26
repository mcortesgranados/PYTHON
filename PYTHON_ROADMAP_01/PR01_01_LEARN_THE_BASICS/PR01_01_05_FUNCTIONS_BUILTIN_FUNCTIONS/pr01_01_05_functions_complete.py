# Python Functions Example

# Example 1: Defining a Simple Function
# Functions in Python are defined using the 'def' keyword followed by the function name and parameters.
# Here's an example of a simple function that prints a greeting message.
def greet():
    """This function prints a greeting message."""
    print("Hello, world!")

# Example 2: Calling a Function
# To call a function, simply use its name followed by parentheses.
# Here's how we call the 'greet' function defined above:
greet()  # Output: Hello, world!

# Example 3: Function Parameters
# Functions can accept parameters, which are variables passed to the function when it's called.
# Here's an example of a function that takes a parameter.
def greet_name(name):
    """This function takes a name as a parameter and prints a greeting message."""
    print("Hello,", name)

# We can call the 'greet_name' function with different names:
greet_name("Alice")  # Output: Hello, Alice
greet_name("Bob")    # Output: Hello, Bob

# Example 4: Default Parameters
# You can specify default values for parameters in a function.
# If the caller doesn't provide a value for that parameter, the default value is used.
def greet_default(name="world"):
    """This function takes an optional 'name' parameter and prints a greeting message."""
    print("Hello,", name)

# Calling the 'greet_default' function with and without specifying a name:
greet_default()       # Output: Hello, world!
greet_default("John") # Output: Hello, John

# Example 5: Returning Values from a Function
# Functions can return values using the 'return' statement.
# Here's an example of a function that calculates the square of a number and returns it.
def square(x):
    """This function calculates the square of a number."""
    return x ** 2

# We can call the 'square' function and store the result in a variable:
result = square(5)
print("Square:", result)  # Output: Square: 25

# Example 6: Multiple Return Values
# Functions can return multiple values as a tuple.
# Here's an example of a function that calculates the square and cube of a number.
def square_and_cube(x):
    """This function calculates the square and cube of a number."""
    return x ** 2, x ** 3

# We can call the 'square_and_cube' function and unpack the returned tuple:
square_result, cube_result = square_and_cube(3)
print("Square:", square_result)  # Output: Square: 9
print("Cube:", cube_result)      # Output: Cube: 27

# Example 7: Variable-Length Arguments (*args)
# Python functions can accept a variable number of arguments using *args.
# This allows you to pass any number of arguments to the function.
# Here's an example of a function that calculates the sum of its arguments.
def sum_all(*args):
    """This function calculates the sum of all its arguments."""
    total = 0
    for num in args:
        total += num
    return total

# We can call the 'sum_all' function with different numbers of arguments:
print("Sum:", sum_all(1, 2, 3, 4))      # Output: Sum: 10
print("Sum:", sum_all(5, 10, 15, 20, 25))# Output: Sum: 75

# Example 8: Keyword Arguments (**kwargs)
# Python functions can also accept keyword arguments using **kwargs.
# This allows you to pass key-value pairs to the function.
# Here's an example of a function that prints key-value pairs.
def print_info(**kwargs):
    """This function prints key-value pairs."""
    for key, value in kwargs.items():
        print(key + ":", value)

# We can call the 'print_info' function with different key-value pairs:
print_info(name="Alice", age=30)  # Output: name: Alice, age: 30
print_info(city="New York", country="USA")  # Output: city: New York, country: USA

# Example 9: Lambda Functions (Anonymous Functions)
# Lambda functions are small anonymous functions defined using the 'lambda' keyword.
# They can have any number of parameters but can only have one expression.
# Here's an example of a lambda function that adds two numbers.
add = lambda x, y: x + y

# We can call the lambda function like a regular function:
print("Sum:", add(2, 3))  # Output: Sum: 5

# Example 10: Recursive Functions
# Recursive functions are functions that call themselves.
# They are useful for solving problems that can be broken down into smaller, similar subproblems.
# Here's an example of a recursive function to calculate the factorial of a number.
def factorial(n):
    """This function calculates the factorial of a number."""
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)

# We can call the 'factorial' function to calculate the factorial of a number:
print("Factorial:", factorial(5))  # Output: Factorial: 120

# Documenting the Functions:
def functions_documentation():
    """
    This function demonstrates various aspects of functions in Python.

    Example 1:
    - Defining a Simple Function: How to define a basic function.

    Example 2:
    - Calling a Function: How to call a function.

    Example 3:
    - Function Parameters: How to define functions with parameters.

    Example 4:
    - Default Parameters: How to specify default values for function parameters.

    Example 5:
    - Returning Values from a Function: How to return values from a function.

    Example 6:
    - Multiple Return Values: How to return multiple values from a function.

    Example 7:
    - Variable-Length Arguments (*args): How to define functions with variable-length arguments.

    Example 8:
    - Keyword Arguments (**kwargs): How to define functions with keyword arguments.

    Example 9:
    - Lambda Functions (Anonymous Functions): How to define anonymous functions using lambda.

    Example 10:
    - Recursive Functions: How to define recursive functions.
    """
    pass

# End of examples
