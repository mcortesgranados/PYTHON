"""
In Python, interfaces are not explicitly defined like in some other programming languages such as Java.
However, the concept of interfaces can be achieved through abstract base classes (ABCs) and duck typing. 
Abstract base classes serve as a blueprint for other classes and define methods that must be implemented 
by concrete subclasses. Duck typing, on the other hand, allows objects to be used based on their 
behavior rather than their type. Below is an example illustrating interfaces in Python using abstract base classes, 
along with detailed explanations and documentation:


"""
from abc import ABC, abstractmethod

# Python Interface Example

class Shape(ABC):
    """
    An abstract base class representing a shape interface.

    Methods:
    - area(self): Calculate the area of the shape.
    - perimeter(self): Calculate the perimeter of the shape.
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

class Rectangle(Shape):
    """
    A concrete class representing a rectangle.

    Attributes:
    - width (float): The width of the rectangle.
    - height (float): The height of the rectangle.
    """

    def __init__(self, width, height):
        """
        Initialize a rectangle with its width and height.

        Args:
        - width (float): The width of the rectangle.
        - height (float): The height of the rectangle.
        """
        self.width = width
        self.height = height

    def area(self):
        """
        Calculate the area of the rectangle.

        Returns:
        - float: The area of the rectangle.
        """
        return self.width * self.height

    def perimeter(self):
        """
        Calculate the perimeter of the rectangle.

        Returns:
        - float: The perimeter of the rectangle.
        """
        return 2 * (self.width + self.height)

# Example: Using Interfaces
# Let's demonstrate the usage of interfaces by creating an instance of the Rectangle class.

# Create a rectangle object
rectangle = Rectangle(5, 4)

# Calculate and display area and perimeter using interface methods
print("Rectangle Area:", rectangle.area())         # Output: 20.0
print("Rectangle Perimeter:", rectangle.perimeter())    # Output: 18.0

# Documenting the Shape and Rectangle Classes:
def interface_documentation():
    """
    This function demonstrates interfaces in Python using abstract base classes.

    Classes:
    - Shape: An abstract base class representing a shape interface.
    - Rectangle: A concrete class representing a rectangle.
    """
    pass

# End of example
