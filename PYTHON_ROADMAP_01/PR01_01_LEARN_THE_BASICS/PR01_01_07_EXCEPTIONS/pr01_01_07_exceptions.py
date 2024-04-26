# This program demonstrates exceptions in Python

# Exceptions (handling unexpected errors)

# Exceptions are events that disrupt the normal flow of your program. 
# Python provides a mechanism to handle these exceptions gracefully 
# using `try...except` blocks. This allows you to write more robust code 
# that can recover from errors and continue execution.

# Common Exception Types:

# - ZeroDivisionError: Attempting to divide by zero
# - TypeError: Trying to perform an operation on incompatible data types
# - ValueError: Providing an inappropriate value for an operation
# - IndexError: Trying to access an element outside a list or sequence bounds
# - KeyError: Accessing a key that doesn't exist in a dictionary
# - FileNotFoundError: Trying to open a file that doesn't exist
# - NameError: Using a variable that hasn't been defined

# Example 1: Handling Multiple Exceptions with Specific Types

try:
  number = int(input("Enter a number: "))  # Get user input (might be invalid)
  if number == 0:
    raise ZeroDivisionError("Custom division by zero message")  # Manually raise exception
  result = 10 / number
  print("Result:", result)
except ZeroDivisionError as e:
  print("Error:", e)  # Access specific error message
except ValueError:
  print("Error: Invalid input. Please enter a number.")

# Example 2: Handling a General Exception (fallback)

try:
  # Code with potential errors (like opening a non-existent file)
  pass  # Placeholder for potentially error-prone code
except Exception as e:  # Catch any unhandled exception
  print("An unexpected error occurred:", e)

# Example 3: Using the `else` Clause (executes if no exception occurs)

try:
  with open("data.txt", "r") as file:  # Open file for reading
    data = file.read()
except FileNotFoundError:
  print("Error: File 'data.txt' not found!")
else:
  print("File data:", data)  # Only executes if no exception happens

# Regardless of errors, the program continues execution (unless critical)

print("Program execution continues...")
