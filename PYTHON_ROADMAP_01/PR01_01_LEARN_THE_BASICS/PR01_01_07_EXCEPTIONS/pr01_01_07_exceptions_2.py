# Python Exceptions Example

# Example 1: Handling Exceptions with try-except block
# You can handle exceptions using a try-except block.
# Code within the try block is executed, and if an exception occurs, it's caught by the except block.
# Here's an example of handling a ZeroDivisionError exception:
try:
    result = 10 / 0  # Attempting to divide by zero
except ZeroDivisionError:
    print("Error: Cannot divide by zero")

# Example 2: Handling Multiple Exceptions
# You can handle multiple exceptions using multiple except blocks or a single except block with a tuple of exceptions.
# Here's an example of handling multiple exceptions:
try:
    result = int("abc")  # Attempting to convert a string to an integer
except (ValueError, TypeError):
    print("Error: Invalid conversion")

# Example 3: Handling Specific Exceptions
# You can handle specific exceptions separately to provide customized error messages or actions.
# Here's an example of handling specific exceptions:
try:
    file = open("nonexistent_file.txt", "r")  # Attempting to open a non-existent file
except FileNotFoundError:
    print("Error: File not found")

# Example 4: Handling All Exceptions with a Generic except Block
# You can use a generic except block to catch any exception that occurs.
# However, it's generally recommended to handle specific exceptions whenever possible.
# Here's an example of using a generic except block:
try:
    result = 10 / 0  # Attempting to divide by zero
except:
    print("An error occurred")

# Example 5: Handling Exceptions with else Block
# You can use an else block in combination with a try-except block.
# The else block is executed if no exception occurs in the try block.
# Here's an example of using an else block:
try:
    result = 10 / 2  # Dividing 10 by 2
except ZeroDivisionError:
    print("Error: Cannot divide by zero")
else:
    print("Result:", result)

# Example 6: Handling Exceptions with finally Block
# You can use a finally block to execute code regardless of whether an exception occurs or not.
# The finally block is executed after the try and except blocks, even if an exception is raised.
# Here's an example of using a finally block:
try:
    file = open("example.txt", "r")  # Attempting to open a file
    print("File opened successfully")
except FileNotFoundError:
    print("Error: File not found")
finally:
    print("Closing file")
    file.close()

# Example 7: Raising Exceptions
# You can raise exceptions explicitly using the raise statement.
# This is useful for signaling errors or exceptional conditions in your code.
# Here's an example of raising an exception:
x = -1
if x < 0:
    raise ValueError("Value must be non-negative")

# Documenting the Exceptions:
def exceptions_documentation():
    """
    This function demonstrates various aspects of exceptions handling in Python.

    Example 1:
    - Handling Exceptions with try-except block: How to catch exceptions using a try-except block.

    Example 2:
    - Handling Multiple Exceptions: How to catch multiple exceptions using multiple except blocks or a single except block with a tuple.

    Example 3:
    - Handling Specific Exceptions: How to handle specific exceptions separately.

    Example 4:
    - Handling All Exceptions with a Generic except Block: How to catch any exception using a generic except block.

    Example 5:
    - Handling Exceptions with else Block: How to execute code if no exception occurs.

    Example 6:
    - Handling Exceptions with finally Block: How to execute code regardless of whether an exception occurs or not.

    Example 7:
    - Raising Exceptions: How to raise exceptions explicitly using the raise statement.
    """
    pass

# End of examples
