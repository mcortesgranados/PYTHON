import datetime

# Code Written by Manuel Cortés Granados
# Date: 2024-03-04
# Location: Bogota DC, Colombia
# LinkedIn Profile: https://www.linkedin.com/in/mcortesgranados/

# Get current date and time
current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# List Comprehensions in Python

# List comprehensions provide a concise way to create lists in Python by applying an expression to each item in an iterable.
# They offer a more readable and expressive alternative to traditional for loops.

# Define a list of numbers
numbers = [1, 2, 3, 4, 5]

# Use a list comprehension to create a new list containing the squares of numbers
squares = [num ** 2 for num in numbers]

# Header for the output
print("Demo code of List Comprehensions in Python")
print("Authored by Manuel Cortés Granados")
print("Date:", current_time)
print("LinkedIn Profile: https://www.linkedin.com/in/mcortesgranados/")
print("-" * 50)  # Separator for a nice display

# Output the original list and the list of squares
print("Original List:", numbers)
print("Squares of Numbers:", squares)

# Output:
# Demo code of List Comprehensions in Python
# Authored by Manuel Cortés Granados
# Date: 2024-03-04 12:30:45
# LinkedIn Profile: https://www.linkedin.com/in/mcortesgranados/
# --------------------------------------------------
# Original List: [1, 2, 3, 4, 5]
# Squares of Numbers: [1, 4, 9, 16, 25]

# List comprehensions can also include conditions to filter the elements of the original iterable.

# Use a list comprehension to create a new list containing only the even numbers from the original list
even_numbers = [num for num in numbers if num % 2 == 0]

# Output the list of even numbers
print("\nEven Numbers:", even_numbers)

# Output:
# Even Numbers: [2, 4]

# List comprehensions provide a concise and readable way to manipulate lists in Python,
# making code more expressive and efficient.
