# Python Class Methods and Static Methods Example

class MathOperations:
    """
    A class representing mathematical operations with class and static methods.

    Class Attributes:
    - PI (float): The mathematical constant pi.

    Class Methods:
    - add(cls, x, y): Add two numbers.
    - multiply(cls, x, y): Multiply two numbers.

    Static Methods:
    - square(x): Calculate the square of a number.
    """

    PI = 3.14  # Class attribute

    @classmethod
    def add(cls, x, y):
        """
        Add two numbers.

        Args:
        - cls (class): The class itself.
        - x (int/float): The first number.
        - y (int/float): The second number.

        Returns:
        - int/float: The sum of the two numbers.
        """
        return x + y

    @classmethod
    def multiply(cls, x, y):
        """
        Multiply two numbers.

        Args:
        - cls (class): The class itself.
        - x (int/float): The first number.
        - y (int/float): The second number.

        Returns:
        - int/float: The product of the two numbers.
        """
        return x * y

    @staticmethod
    def square(x):
        """
        Calculate the square of a number.

        Args:
        - x (int/float): The number to be squared.

        Returns:
        - int/float: The square of the number.
        """
        return x ** 2

# Example: Using Class Methods and Static Methods
# Let's demonstrate the usage of class methods and static methods by calling them directly from the class.

# Class methods
print("Sum:", MathOperations.add(5, 3))          # Output: 8
print("Product:", MathOperations.multiply(5, 3))  # Output: 15

# Static method
print("Square:", MathOperations.square(5))        # Output: 25

# Documenting the MathOperations Class:
def class_static_methods_documentation():
    """
    This function demonstrates class methods and static methods in Python.

    Classes:
    - MathOperations: A class representing mathematical operations with class and static methods.
    """
    pass

# End of example
