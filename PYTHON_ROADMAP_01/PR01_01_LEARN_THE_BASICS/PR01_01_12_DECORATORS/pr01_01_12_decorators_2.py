# Python Decorators Example

# Example 1: Creating a Simple Decorator
# A decorator is a function that takes another function as an argument and returns a new function.
# It allows you to add functionality to existing functions without modifying their code.
# Here's an example of creating a simple decorator that adds logging to a function:

def logger(func):
    """A decorator that adds logging to a function."""
    def wrapper(*args, **kwargs):
        print(f"Calling function: {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

@logger
def add(a, b):
    """Add two numbers."""
    return a + b

result = add(3, 5)
print("Result:", result)

# Example 2: Decorating Functions with Arguments
# Decorators can also accept arguments.
# You can achieve this by defining a decorator function that returns another function (the actual decorator).
# Here's an example of creating a decorator with arguments to specify the log level:

def logger_with_level(level):
    """A decorator that adds logging with specified log level to a function."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(f"({level}) Calling function: {func.__name__}")
            return func(*args, **kwargs)
        return wrapper
    return decorator

@logger_with_level("INFO")
def multiply(a, b):
    """Multiply two numbers."""
    return a * b

result = multiply(3, 5)
print("Result:", result)

# Example 3: Decorating Methods in Classes
# Decorators can also be used to modify the behavior of methods in classes.
# Here's an example of creating a class with a method decorated with a logger:

class Calculator:
    """A simple calculator class."""

    @logger
    def add(self, a, b):
        """Add two numbers."""
        return a + b

calc = Calculator()
result = calc.add(3, 5)
print("Result (Method Decoration):", result)

# Example 4: Using functools.wraps for Preserving Metadata
# When creating decorators, it's important to preserve the metadata of the original function (like docstrings and function name).
# You can achieve this using the functools.wraps decorator.
# Here's an example of using functools.wraps to preserve metadata:

import functools

def logger_preserve_metadata(func):
    """A decorator that preserves metadata of the original function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Calling function: {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

@logger_preserve_metadata
def subtract(a, b):
    """Subtract two numbers."""
    return a - b

result = subtract(8, 3)
print("Result (Preserve Metadata):", result)
print("Docstring (Preserve Metadata):", subtract.__doc__)

# Documenting the Decorators:
def decorators_documentation():
    """
    This function demonstrates various aspects of decorators in Python.

    Example 1:
    - Creating a Simple Decorator: How to create a simple decorator to add logging to a function.

    Example 2:
    - Decorating Functions with Arguments: How to create a decorator with arguments.

    Example 3:
    - Decorating Methods in Classes: How to use decorators to modify the behavior of methods in classes.

    Example 4:
    - Using functools.wraps for Preserving Metadata: How to preserve metadata of the original function using functools.wraps.
    """
    pass

# End of examples
