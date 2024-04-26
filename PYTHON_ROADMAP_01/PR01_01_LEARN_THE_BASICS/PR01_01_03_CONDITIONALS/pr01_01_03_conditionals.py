# This program checks if a number is even or odd

# Get a number from the user
number = int(input("Enter a number: "))

# Check if the number is even using a conditional statement
if number % 2 == 0:
  # Code block to execute if the condition is True
  print(number, "is an even number.")
else:
  # Code block to execute if the condition is False
  print(number, "is an odd number.")
