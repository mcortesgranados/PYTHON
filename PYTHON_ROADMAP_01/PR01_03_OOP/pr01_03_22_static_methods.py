"""
Static methods in Python are methods that belong to a class but do not operate on instances or class attributes. 
They are independent of both instances and classes and are defined using the @staticmethod decorator. 
Static methods are primarily used for utility functions that do not require access to instance or class attributes. 
Below is an example illustrating static methods in Python, along with detailed explanations and documentation:

"""

# Python Static Methods Example

class MathOperations:
    """
    A class demonstrating static methods in Python for performing mathematical operations.

    Methods:
    - add(x, y): A static method to perform addition.
    - subtract(x, y): A static method to perform subtraction.
    """

    @staticmethod
    def add(x, y):
        """
        Perform addition of two numbers.

        Args:
        - x (int/float): The first number.
        - y (int/float): The second number.

        Returns:
        - int/float: The result of addition.
        """
        return x + y

    @staticmethod
    def subtract(x, y):
        """
        Perform subtraction of two numbers.

        Args:
        - x (int/float): The first number.
        - y (int/float): The second number.

        Returns:
        - int/float: The result of subtraction.
        """
        return x - y

# Example: Using Static Methods
# Let's demonstrate the usage of static methods by performing mathematical operations.

# Perform addition using static method
result_addition = MathOperations.add(5, 3)
print("Result of Addition:", result_addition)  # Output: 8

# Perform subtraction using static method
result_subtraction = MathOperations.subtract(10, 4)
print("Result of Subtraction:", result_subtraction)  # Output: 6

# Documenting the MathOperations Class:
def static_methods_documentation():
    """
    This function demonstrates static methods in Python.

    Classes:
    - MathOperations: A class demonstrating static methods in Python for performing mathematical operations.
    """
    pass

# End of example
