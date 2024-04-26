# Python Example: Demonstrating Multiple Programming Paradigms

# Procedural Paradigm:
# Procedural programming focuses on writing procedures or routines to perform tasks.
# Functions are the building blocks of procedural programming.
# Here's an example of procedural programming to calculate the factorial of a number:

def factorial_procedural(n):
    """Calculate the factorial of a number using procedural paradigm."""
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

print("Factorial (Procedural):", factorial_procedural(5))


# Object-Oriented Paradigm:
# Object-oriented programming (OOP) focuses on creating objects that encapsulate data and behavior.
# Classes and objects are central concepts in OOP.
# Here's an example of object-oriented programming using a class to represent a rectangle:

class Rectangle:
    """A class representing a rectangle."""

    def __init__(self, width, height):
        """Initialize the rectangle with width and height."""
        self.width = width
        self.height = height

    def area(self):
        """Calculate the area of the rectangle."""
        return self.width * self.height

    def perimeter(self):
        """Calculate the perimeter of the rectangle."""
        return 2 * (self.width + self.height)

rectangle = Rectangle(4, 5)
print("Area (Object-Oriented):", rectangle.area())
print("Perimeter (Object-Oriented):", rectangle.perimeter())


# Functional Paradigm:
# Functional programming emphasizes the use of pure functions and avoids mutable data.
# Functions are first-class citizens, meaning they can be passed around and returned from other functions.
# Here's an example of functional programming using lambda functions and map() to calculate the squares of numbers:

numbers = [1, 2, 3, 4, 5]

squares_functional = list(map(lambda x: x**2, numbers))
print("Squares (Functional):", squares_functional)

# Documenting the Example:
def paradigms_example():
    """
    This function demonstrates multiple programming paradigms in Python.

    Procedural Paradigm:
    - Calculate the factorial of a number using procedural programming.

    Object-Oriented Paradigm:
    - Create a rectangle object and calculate its area and perimeter using object-oriented programming.

    Functional Paradigm:
    - Calculate the squares of numbers using functional programming with lambda functions and map().
    """
    pass

# End of example
