"""
Paradigms in Python (Comprehensive Demonstration)

This script demonstrates various programming paradigms supported by Python:

* Imperative Programming
* Functional Programming
* Object-Oriented Programming

This example demonstrates three common paradigms in Python:

Imperative Programming:
Represented by a function factorial_imperative that uses a loop to calculate the factorial iteratively.
Focuses on the "how" of achieving the task (steps involved).
Functional Programming:
Represented by a function factorial_functional that uses recursion to calculate the factorial.
Focuses on the "what" of achieving the task (the result without specifying steps).
Leverages a built-in function for a concise implementation.
Object-Oriented Programming:
Represented by a class Point that defines a Point object with attributes (x, y) and a method (distance_to_origin).
Organizes code around objects and their interactions.
This script provides comments for clarity and concludes with a summary of the key points for each paradigm.
"""

# Imperative Programming (focusing on how to achieve a task)

# Define a function to calculate the factorial of a number
def factorial_imperative(n):
  """Calculates the factorial of a number using an iterative approach."""
  result = 1
  for i in range(1, n + 1):
    result *= i
  return result

# Call the function to calculate factorial of 5
factorial = factorial_imperative(5)
print("Factorial of 5 (imperative):", factorial)

# Functional Programming (focusing on what to achieve)

# Use a built-in function (recursion) for factorial
def factorial_functional(n):
  """Calculates the factorial of a number using a recursive approach."""
  if n == 0:
    return 1
  else:
    return n * factorial_functional(n - 1)

# Call the function to calculate factorial of 5
factorial = factorial_functional(5)
print("Factorial of 5 (functional):", factorial)

# Object-Oriented Programming (focusing on data and its behavior)

# Define a class to represent a Point
class Point:
  """Represents a point in 2D space with x and y coordinates."""

  def __init__(self, x, y):
    """Initialize the Point object with x and y coordinates."""
    self.x = x
    self.y = y

  def distance_to_origin(self):
    """Calculates the distance of the point from the origin."""
    return (self.x**2 + self.y**2)**0.5

# Create a Point object
point = Point(3, 4)

# Access object attributes and call methods
print("Distance from origin:", point.distance_to_origin())

# Key Points:

* Imperative programming focuses on the sequence of steps to achieve a task.
* Functional programming emphasizes pure functions and immutability of data.
* Object-oriented programming organizes code around objects and their interactions.

"""This script concludes the demonstration of programming paradigms."""
