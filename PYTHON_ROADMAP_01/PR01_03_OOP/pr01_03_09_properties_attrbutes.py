# Python Properties and Attributes Example

class Rectangle:
    """
    A class representing a rectangle with properties and attributes.

    Attributes:
    - _width (float): The width of the rectangle.
    - _height (float): The height of the rectangle.
    """

    def __init__(self, width, height):
        """
        Initialize a rectangle with its width and height.

        Args:
        - width (float): The width of the rectangle.
        - height (float): The height of the rectangle.
        """
        self._width = width
        self._height = height

    @property
    def width(self):
        """
        Get the width of the rectangle.

        Returns:
        - float: The width of the rectangle.
        """
        return self._width

    @width.setter
    def width(self, value):
        """
        Set the width of the rectangle.

        Args:
        - value (float): The new width of the rectangle.
        """
        if value <= 0:
            raise ValueError("Width must be positive")
        else:
            self._width = value

    @property
    def height(self):
        """
        Get the height of the rectangle.

        Returns:
        - float: The height of the rectangle.
        """
        return self._height

    @height.setter
    def height(self, value):
        """
        Set the height of the rectangle.

        Args:
        - value (float): The new height of the rectangle.
        """
        if value <= 0:
            raise ValueError("Height must be positive")
        else:
            self._height = value

    def area(self):
        """
        Calculate the area of the rectangle.

        Returns:
        - float: The area of the rectangle.
        """
        return self._width * self._height

    def perimeter(self):
        """
        Calculate the perimeter of the rectangle.

        Returns:
        - float: The perimeter of the rectangle.
        """
        return 2 * (self._width + self._height)

# Example: Using Properties and Attributes
# Let's demonstrate the usage of properties and attributes by creating an instance of the Rectangle class.

# Create a rectangle object
rectangle = Rectangle(4, 5)

# Get and set width and height using properties
print("Initial Width:", rectangle.width)  # Output: 4
print("Initial Height:", rectangle.height)  # Output: 5

rectangle.width = 6
rectangle.height = 7

print("Updated Width:", rectangle.width)  # Output: 6
print("Updated Height:", rectangle.height)  # Output: 7

# Calculate area and perimeter using methods
print("Area:", rectangle.area())        # Output: 42
print("Perimeter:", rectangle.perimeter())  # Output: 26

# Documenting the Rectangle Class:
def properties_attributes_documentation():
    """
    This function demonstrates properties and attributes in Python.

    Classes:
    - Rectangle: A class representing a rectangle with properties and attributes.
    """
    pass

# End of example
