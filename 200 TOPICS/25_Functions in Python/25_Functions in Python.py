# Functions in Python are blocks of reusable code that perform a specific task. Below is a Python code sample demonstrating the creation and usage of functions:
# In this code:

# We define a function greet() using the def keyword, followed by the function name and parameter(s) in parentheses.
# The function body is indented, and it contains the code that performs the desired task.
# Inside the function, we use the print() function to output a greeting message.
# We call the function multiple times with different arguments to greet different people.
# Functions can accept parameters (also called arguments) and return values. They allow you to organize your code into logical blocks, 
# making it easier to read, understand, and maintain. The docstring enclosed in triple quotes """ """ provides documentation for the function, 
# describing its purpose and usage.

# Define a function
def greet(name):
    """This function greets the person with the given name."""
    print("Hello,", name)

# Call the function
greet("Alice")
greet("Bob")
