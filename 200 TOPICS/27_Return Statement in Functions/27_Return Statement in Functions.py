# In this code:

# The add() function takes two parameters a and b and returns their sum using the return statement.
# The absolute_value() function takes a single parameter x and returns its absolute value using conditional statements and multiple return statements.
# When a function encounters a return statement, it immediately exits the function and returns the specified value to the caller.
# The return statement is used to send a value back to the caller of the function. It allows functions to produce output that can be used elsewhere in the code.

# Function with a return statement
def add(a, b):
    """This function returns the sum of two numbers."""
    return a + b

# Call the function and store the result
result = add(3, 5)
print("Result of addition:", result)

# Function with multiple return statements
def absolute_value(x):
    """This function returns the absolute value of a number."""
    if x >= 0:
        return x
    else:
        return -x

# Call the function and store the result
absolute_result = absolute_value(-7)
print("Absolute value:", absolute_result)
