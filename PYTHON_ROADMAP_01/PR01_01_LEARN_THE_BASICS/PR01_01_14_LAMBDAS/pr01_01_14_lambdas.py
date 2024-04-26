# Lambda Function Example

# Define a list of numbers
numbers = [1, 2, 3, 4, 5]

# Example 1: Basic Lambda Function
# Lambda functions are anonymous functions that can have any number of arguments, but only one expression.
# Syntax: lambda arguments: expression
# Here, we define a lambda function to square a number.
square = lambda x: x ** 2

# Example 2: Using Lambdas with Built-in Functions
# Lambdas can be used as arguments in built-in functions like map(), filter(), and reduce().
# Here, we use map() to square each number in the list.
squared_numbers = list(map(lambda x: x ** 2, numbers))

# Example 3: Using Lambdas with Filter
# We use filter() to select only even numbers from the list.
even_numbers = list(filter(lambda x: x % 2 == 0, numbers))

# Example 4: Using Lambdas with Sorting
# Lambdas can be used to define custom sorting criteria.
# Here, we sort a list of tuples based on the second element of each tuple.
tuple_list = [(1, 'apple'), (3, 'banana'), (2, 'orange')]
sorted_tuples = sorted(tuple_list, key=lambda x: x[1])

# Example 5: Using Lambdas with Key Argument in min() and max()
# We use min() and max() functions with a custom key to find the shortest and longest string in a list.
strings = ['apple', 'banana', 'orange']
shortest_string = min(strings, key=lambda x: len(x))
longest_string = max(strings, key=lambda x: len(x))

# Example 6: Using Lambdas with Reduce (from functools)
# Reduce applies a rolling computation to sequential pairs of values in a list.
# Here, we use it to find the product of all numbers in the list.
from functools import reduce
product = reduce(lambda x, y: x * y, numbers)

# Documenting the Lambda Function:
def lambda_documentation():
    """
    This function demonstrates various use cases of lambda functions in Python.

    Example 1:
    - Basic lambda function to square a number.

    Example 2:
    - Using lambdas with built-in functions like map().

    Example 3:
    - Using lambdas with filter() to select even numbers from a list.

    Example 4:
    - Using lambdas for custom sorting criteria with sorted().

    Example 5:
    - Using lambdas with key argument in min() and max() to find shortest and longest strings.

    Example 6:
    - Using lambdas with reduce() to find the product of all numbers in a list.
    """
    pass

# Example 7: Using Lambdas with Conditional Expressions
# Lambda functions can also include conditional expressions.
# Here, we define a lambda function to return 'Even' or 'Odd' based on the number.
even_or_odd = lambda x: 'Even' if x % 2 == 0 else 'Odd'

# Example 8: Using Lambdas with Default Arguments
# Lambda functions can't have default arguments. But, you can achieve similar functionality using closures.
def power_n(n):
    return lambda x: x ** n

square = power_n(2)  # Create a function to square a number
cube = power_n(3)    # Create a function to cube a number

# Example 9: Using Lambdas with Higher-Order Functions
# Lambda functions can be passed to other functions as arguments.
def apply_operation(operation, x, y):
    return operation(x, y)

# We define a lambda function for addition and pass it to apply_operation().
result = apply_operation(lambda a, b: a + b, 10, 5)

# Example 10: Using Lambdas in GUI Programming (Tkinter)
# Lambdas are commonly used in GUI programming to define event handlers concisely.
import tkinter as tk

root = tk.Tk()
button = tk.Button(root, text="Click me", command=lambda: print("Button clicked!"))
button.pack()

# Run the Tkinter event loop
root.mainloop()

# Example 11: Using Lambdas for Delayed Execution
# Lambdas can be used to create functions with pre-defined arguments for delayed execution.
def delayed_execution(func, *args):
    return lambda: func(*args)

# Define a function to greet someone
def greet(name):
    print(f"Hello, {name}!")

# Create a delayed execution function to greet 'World'
greet_world = delayed_execution(greet, 'World')

# Example 12: Using Lambdas for Data Transformation
# Lambdas are useful for transforming data, such as converting Celsius to Fahrenheit.
celsius_temperatures = [0, 10, 20, 30, 40]
fahrenheit_temperatures = list(map(lambda c: (c * 9/5) + 32, celsius_temperatures))

# Example 13: Using Lambdas for Function Composition
# Lambdas can be composed to create more complex functions.
def compose(func1, func2):
    return lambda x: func1(func2(x))

# Define two simple functions
add_one = lambda x: x + 1
multiply_by_two = lambda x: x * 2

# Compose the functions to create a new function
add_one_then_multiply_by_two = compose(multiply_by_two, add_one)

# Test the composed function
result = add_one_then_multiply_by_two(3)  # Should return 8 (3 + 1 = 4, 4 * 2 = 8)

# Example 14: Using Lambdas for Currying
# Currying is the technique of translating the evaluation of a function that takes multiple arguments
# into evaluating a sequence of functions, each with a single argument.
def curry(func):
    def curried(*args):
        if len(args) >= func.__code__.co_argcount:
            return func(*args)
        return lambda *next_args: curried(*(args + next_args))
    return curried

# Define a function that adds three numbers
def add_three(x, y, z):
    return x + y + z

# Curry the function
curried_add_three = curry(add_three)

# Test the curried function
result = curried_add_three(1)(2)(3)  # Should return 6 (1 + 2 + 3 = 6)

# Example 15: Using Lambdas for Memoization
# Memoization is an optimization technique used primarily to speed up computer programs by storing the results of expensive function calls.
def memoize(func):
    cache = {}

    def memoized_func(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]

    return memoized_func

# Define a slow function that computes Fibonacci numbers recursively
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# Memoize the Fibonacci function
memoized_fibonacci = memoize(fibonacci)

# Test the memoized function
result = memoized_fibonacci(10)  # Should return 55 (fibonacci(10) = 55)

# Example 16: Using Lambdas with Regular Expressions
# Lambdas can be used with regular expressions for custom matching logic.
import re

# Define a list of strings
strings = ['apple', 'banana', 'orange', 'pineapple']

# Use filter() with a lambda to find strings containing 'apple'
filtered_strings = list(filter(lambda x: re.search(r'apple', x), strings))

# Print the filtered strings
print(filtered_strings)  # Output: ['apple', 'pineapple']

# End of examples
