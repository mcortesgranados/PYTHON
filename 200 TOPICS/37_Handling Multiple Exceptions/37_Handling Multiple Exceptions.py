# In the first try block:

# We attempt to perform a division operation (10 / x) after converting user input to an integer (int(input("Enter a number: "))).
# If either a ValueError (invalid input) or a ZeroDivisionError occurs during the execution of the try block, the program jumps to the corresponding except block, where we handle the errors by printing a custom error message.
# In the second try block:

# We attempt to open a non-existent file 'nonexistent_file.txt' for reading.
# If a FileNotFoundError occurs during the execution of the try block, the program jumps to the corresponding except block, 
# where we handle the error by printing a custom error message.
# We also demonstrate catching a PermissionError for cases where the user may not have permission to access the file.
# We use a general except block to catch any other exceptions that might occur and print the error message.
# Finally, we ensure that the file is closed in the finally block, regardless of whether an exception occurred or not.
# Handling multiple exceptions allows your program to react appropriately to different types of errors and ensures that it 
# continues to function even in unexpected scenarios.

# Handling multiple exceptions in a single except block
try:
    x = int(input("Enter a number: "))
    result = 10 / x
    print("Result of division:", result)
except (ValueError, ZeroDivisionError):
    print("Error: Invalid input or division by zero!")

# Handling multiple exceptions with separate except blocks
try:
    file = open('nonexistent_file.txt', 'r')
    content = file.read()
    print("Content of the file:", content)
except FileNotFoundError:
    print("Error: File not found!")
except PermissionError:
    print("Error: Permission denied!")
except Exception as e:
    print("Error:", e)
finally:
    if 'file' in locals():
        file.close()
        print("File closed successfully.")
