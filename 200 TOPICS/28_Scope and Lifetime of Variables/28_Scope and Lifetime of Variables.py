# 28_Scope and Lifetime of Variables
# In Python, the scope of a variable refers to the region of the program where the variable is accessible. The lifetime of a variable refers to the period during which the variable exists in memory. Below is a Python code sample demonstrating the scope and lifetime of variables:

# In this code:

# global_var is a global variable, defined outside any function. It is accessible from anywhere in the code.
# local_var is a local variable, defined inside the my_function() function. It is accessible only within the function.
# Inside the function, both global and local variables can be accessed.
# Outside the function, only the global variable can be accessed.
# Attempting to access the local variable outside the function will result in a NameError because it is not defined in the global scope.
# The global variable exists throughout the program's execution, whereas the local variable exists only within the function's execution and is destroyed once the function completes its execution. This demonstrates the scope and lifetime of variables in Python.

# Global variable
global_var = 10

# Function definition
def my_function():
    # Local variable
    local_var = 20
    print("Inside function: global_var =", global_var)  # Accessing global variable
    print("Inside function: local_var =", local_var)

# Call the function
my_function()

# Accessing global variable outside the function
print("Outside function: global_var =", global_var)

# Attempting to access local variable outside the function (will raise NameError)
# print("Outside function: local_var =", local_var)

