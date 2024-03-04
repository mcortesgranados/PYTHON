import datetime

# Code Written by Manuel Cortés Granados
# Date: 2024-03-04
# Location: Bogota DC, Colombia

# Get current date and time
current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Decorators allow you to extend and modify the behavior of callable objects (functions, methods, or classes)
# without permanently modifying their source code. They are implemented as functions that wrap other functions.

# Define a decorator function
def my_decorator(func):
    """
    This decorator function takes another function as an argument and returns a new function that 
    wraps the original function, providing additional functionality before and after calling it.
    """
    # Define a wrapper function inside the decorator
    def wrapper():
        # Actions to be performed before calling the original function
        print("Something is happening before the function is called.")
        
        # Call the original function
        func()  # Call the original function
        
        # Actions to be performed after calling the original function
        print("Something is happening after the function is called.")
    
    # Return the wrapper function
    return wrapper

# Define a function to be decorated
def say_hello():
    """
    This function prints 'Hello!' to the console.
    """
    print("Hello!")

# Decorate the function by assigning the result of calling my_decorator with say_hello as an argument to say_hello
say_hello = my_decorator(say_hello)

# Call the decorated function
say_hello()

# Print the current time
print("Current Time:", current_time)
