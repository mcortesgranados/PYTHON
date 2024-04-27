# Python Abstraction Example

from abc import ABC, abstractmethod

class Shape(ABC):
    """
    A base class representing a geometric shape.

    Methods:
    - area(): Calculate the area of the shape.
    - perimeter(): Calculate the perimeter of the shape.
    """

    @abstractmethod
    def area(self):
        """
        Abstract method to calculate the area of the shape.
        """
        pass

    @abstractmethod
    def perimeter(self):
        """
        Abstract method to calculate the perimeter of the shape.
        """
        pass

class Circle(Shape):
    """
    A class representing a circle, inheriting from the Shape class.

    Attributes:
    - radius (float): The radius of the circle.
    """

    def __init__(self, radius):
        """
        Initialize a circle with its radius.

        Args:
        - radius (float): The radius of the circle.
        """
        self.radius = radius

    def area(self):
        """
        Calculate the area of the circle.

        Returns:
        - float: The area of the circle.
        """
        return 3.14 * self.radius ** 2

    def perimeter(self):
        """
        Calculate the perimeter of the circle.

        Returns:
        - float: The perimeter of the circle.
        """
        return 2 * 3.14 * self.radius

class Square(Shape):
    """
    A class representing a square, inheriting from the Shape class.

    Attributes:
    - side_length (float): The length of a side of the square.
    """

    def __init__(self, side_length):
        """
        Initialize a square with its side length.

        Args:
        - side_length (float): The length of a side of the square.
        """
        self.side_length = side_length

    def area(self):
        """
        Calculate the area of the square.

        Returns:
        - float: The area of the square.
        """
        return self.side_length ** 2

    def perimeter(self):
        """
        Calculate the perimeter of the square.

        Returns:
        - float: The perimeter of the square.
        """
        return 4 * self.side_length

# Example: Using Abstraction
# Let's demonstrate the usage of abstraction by creating instances of the Circle and Square classes.

# Create shape objects
circle = Circle(5)
square = Square(4)

# Calculate area and perimeter using the same interface
print("Circle Area:", circle.area())      # Output: 78.5
print("Circle Perimeter:", circle.perimeter())  # Output: 31.400000000000002
print("Square Area:", square.area())      # Output: 16
print("Square Perimeter:", square.perimeter())  # Output: 16

# Documenting the Shape, Circle, and Square Classes:
def abstraction_documentation():
    """
    This function demonstrates abstraction in Python.

    Classes:
    - Shape: A base class representing a geometric shape.
    - Circle: A class representing a circle, inheriting from the Shape class.
    - Square: A class representing a square, inheriting from the Shape class.
    """
    pass

# End of example
