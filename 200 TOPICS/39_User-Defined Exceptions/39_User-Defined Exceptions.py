# In this code:

# We define a custom exception class named MyCustomException, which inherits from the built-in Exception class. 
# The constructor (__init__ method) accepts an optional message parameter, which defaults to a custom error message.
# We define a function named divide(x, y) that takes two arguments x and y. If y is zero, we raise a MyCustomException with a custom error message.
# We use a try block to call the divide() function with arguments 10 and 0. If a MyCustomException is raised during the execution of the try block, 
# the program jumps to the corresponding except block, where we catch the exception and print the error message.
# Defining and using custom exceptions allows you to create more meaningful and descriptive error handling in your Python programs. 
# It also helps in organizing and managing error handling logic effectively.

# Define a custom exception class
class MyCustomException(Exception):
    def __init__(self, message="This is a custom exception."):
        self.message = message
        super().__init__(self.message)

# Function that raises the custom exception
def divide(x, y):
    if y == 0:
        raise MyCustomException("Division by zero is not allowed.")
    return x / y

# Example usage
try:
    divide(10, 0)
except MyCustomException as e:
    print("Custom exception caught:", e)
