# In this code:

#We attempt to perform a division operation (10 / x) after converting user input to an integer (int(input("Enter a number: "))).
# We use separate except blocks to handle specific exceptions:
# If a ValueError occurs during the execution of the try block (e.g., if the user enters a non-integer value), the program 
# jumps to the corresponding except block, where we handle the error by printing a custom error message.
# If a ZeroDivisionError occurs during the execution of the try block (e.g., if the user enters zero), 
# the program jumps to the corresponding except block, where we handle the error by printing a custom error message.
# We use a general except block to catch any other exceptions that might occur and print the error message. This can be helpful 
# for debugging purposes or for catching unexpected errors.
# Handling specific exceptions allows your program to respond appropriately to different types of errors, improving its robustness and user experience.

 # Handling specific exceptions with separate except blocks
try:
    x = int(input("Enter a number: "))
    result = 10 / x
    print("Result of division:", result)
except ValueError:
    print("Error: Invalid input! Please enter a valid number.")
except ZeroDivisionError:
    print("Error: Division by zero is not allowed!")
except Exception as e:
    print("An error occurred:", e)

