# In this code:

# We attempt to perform a division operation (10 / 0) inside a try block.
# If a ZeroDivisionError occurs during the execution of the try block, the program jumps to the corresponding except block, 
# where we handle the error gracefully by printing a custom error message.
# We then attempt to open a non-existent file 'nonexistent_file.txt' for reading inside another try block.
# If a FileNotFoundError occurs during the execution of the second try block, the program jumps to the corresponding except block, 
# where we handle the error by printing a custom error message.
# We use a finally block to ensure that the file is closed regardless of whether an exception occurred or not. 
# This block is executed irrespective of whether an exception was raised or not.
# Exception handling helps make Python programs more robust by allowing them to handle unexpected errors gracefully without crashing. 
# The try, except, and finally blocks provide a structured way to handle exceptions and perform cleanup operations.

# Division by zero error
try:
    result = 10 / 0
    print("Result of division:", result)
except ZeroDivisionError:
    print("Error: Division by zero!")

# File handling error
try:
    file = open('nonexistent_file.txt', 'r')
    content = file.read()
    print("Content of the file:", content)
except FileNotFoundError:
    print("Error: File not found!")
finally:
    if 'file' in locals():
        file.close()
        print("File closed successfully.")
