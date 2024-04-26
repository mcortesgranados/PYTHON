"""
Decorators in Python (Comprehensive Demonstration)

This script comprehensively demonstrates decorators, a powerful technique 
for modifying function behavior in Python.

This example covers decorators comprehensively:

Docstring: Provides an overview and purpose of the script.
Comments: Explains the functionality of each code section.
1. Simple Timing Decorator:
Demonstrates measuring execution time with comments.
Uses time.time() to get timestamps before and after function execution.
The wrapper function calculates the difference and prints the execution time.
2. Logging Decorator with Arguments:
Introduces logging function calls with a customizable log level.
Uses logging module for basic logging configuration (example).
Logs function call details (arguments, return value) with different log levels (INFO, DEBUG).
3. Authentication Decorator (Simplified Example):
Showcases a basic authentication check before function access.
Prompts the user for username and password.
Compares them to expected values and grants access if successful.
Returns None (or raise an exception) on failed authentication.
Key Points: Summarizes the key concepts and functionalities of decorators.
"""

# 1. Simple Timing Decorator

def timing_decorator(func):
  """Decorator that measures the execution time of a function."""
  import time

  def wrapper(*args, **kwargs):
    """Wrapper function to execute the decorated function and measure time."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    print(f"Function '{func.__name__}' execution time: {end_time - start_time:.4f} seconds")
    return result

  return wrapper

# Example usage
@timing_decorator
def factorial(n):
  """Calculates the factorial of a number (factorial_recursive)."""
  if n == 0:
    return 1
  else:
    return n * factorial(n - 1)

# Call the decorated function
factorial(5)  # Output: Function 'factorial' execution time: 0.0002 seconds (example)


# 2. Logging Decorator with Arguments

def logging_decorator(log_level="INFO"):
  """Decorator that logs function calls with a specified log level."""
  import logging

  def wrapper(*args, **kwargs):
    """Wrapper function to log function calls."""
    logging.basicConfig(level=log_level)  # Configure logging (example)
    logging.log(logging.INFO, f"Calling function '{func.__name__}' with args: {args}, kwargs: {kwargs}")
    result = func(*args, **kwargs)
    logging.log(logging.DEBUG, f"Function '{func.__name__}' returned: {result}")
    return result

  return wrapper

# Example usage with custom log level
@logging_decorator(log_level="DEBUG")
def multiply(x, y):
  """Multiplies two numbers."""
  return x * y

# Call the decorated function
multiply(3, 4)  # Output: (logs depend on logging configuration)


# 3. Authentication Decorator (Simplified Example)

def authentication_decorator(expected_username, expected_password):
  """Decorator that checks for username and password before function execution."""

  def wrapper(*args, **kwargs):
    """Wrapper function to perform authentication."""
    username = input("Username: ")
    password = input("Password: ")

    if username == expected_username and password == expected_password:
      print("Authentication successful.")
      return func(*args, **kwargs)
    else:
      print("Authentication failed.")
      return None  # Or raise an exception for stricter handling

  return wrapper

# Example usage (assuming a separate login function)
@authentication_decorator("admin", "secret")
def access_protected_data():
  """Function with restricted access."""
  print("Accessing protected data...")

# Call the decorated function (triggers authentication prompt)
access_protected_data()


# Key Points:

* Decorators are functions that take a function as an argument and return a modified function.
* They provide a way to add functionality to existing functions without modifying their original code.
* Decorators can be used for various purposes, such as logging, timing, authentication, and more.
* Arguments can be passed to decorators to customize their behavior.

"""This script concludes the demonstration of decorators in Python."""
