import datetime

# Code Written by Manuel Cortés Granados
# Date: 2024-03-04
# Location: Bogota DC, Colombia
# LinkedIn Profile: https://www.linkedin.com/in/mcortesgranados/

# Get current date and time
current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Lambda Functions in Python

# Lambda functions, also known as anonymous functions, allow you to create small, unnamed functions on-the-fly.
# They are defined using the 'lambda' keyword, followed by a list of parameters and an expression.

# Define a lambda function to compute the square of a number
square = lambda x: x ** 2

# Header for the output
print("Demo code of Lambda Functions in Python")
print("Authored by Manuel Cortés Granados")
print("Date:", current_time)
print("LinkedIn Profile: https://www.linkedin.com/in/mcortesgranados/")
print("-" * 50)  # Separator for a nice display

# Use the lambda function to compute squares
print("Square of 5:", square(5))
print("Square of 10:", square(10))

# Output:
# Demo code of Lambda Functions in Python
# Authored by Manuel Cortés Granados
# Date: 2024-03-04 12:30:45
# LinkedIn Profile: https://www.linkedin.com/in/mcortesgranados/
# --------------------------------------------------
# Square of 5: 25
# Square of 10: 100

# Lambda functions can also be used in combination with built-in functions like map(), filter(), and reduce().

# Use the map() function with a lambda function to compute squares of numbers in a list
numbers = [1, 2, 3, 4, 5]
squared_numbers = list(map(lambda x: x ** 2, numbers))

# Output the list of squared numbers
print("\nSquared Numbers:", squared_numbers)

# Output:
# Squared Numbers: [1, 4, 9, 16, 25]

# Lambda functions provide a concise way to define simple functions inline,
# making code more readable and expressive.
