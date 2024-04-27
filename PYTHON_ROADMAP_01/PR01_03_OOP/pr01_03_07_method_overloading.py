# Python Method Overloading Example

class Calculator:
    """
    A class representing a simple calculator with method overloading.

    Methods:
    - add(x, y): Add two numbers.
    """

    def add(self, x, y=None):
        """
        Add two numbers. If only one argument is provided, add it to itself.

        Args:
        - x (int/float): The first number.
        - y (int/float, optional): The second number. Default is None.
        """
        if y is None:
            return x + x
        else:
            return x + y

# Example: Using Method Overloading
# Let's demonstrate the usage of method overloading by creating an instance of the Calculator class.

# Create a calculator object
calculator = Calculator()

# Add two numbers using different argument combinations
print("Result 1:", calculator.add(5, 3))    # Output: 8
print("Result 2:", calculator.add(5))       # Output: 10 (5 + 5)

# Documenting the Calculator Class:
def method_overloading_documentation():
    """
    This function demonstrates method overloading in Python using default parameter values.

    Classes:
    - Calculator: A class representing a simple calculator with method overloading.
    """
    pass

# End of example
