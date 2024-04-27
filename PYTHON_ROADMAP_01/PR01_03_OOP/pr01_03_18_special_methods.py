# Python Special Methods (Magic Methods) Example

class Vector:
    """
    A class representing a vector in 2D space.

    Attributes:
    - x (float): The x-coordinate of the vector.
    - y (float): The y-coordinate of the vector.
    """

    def __init__(self, x, y):
        """
        Initialize a vector with its x and y coordinates.

        Args:
        - x (float): The x-coordinate of the vector.
        - y (float): The y-coordinate of the vector.
        """
        self.x = x
        self.y = y

    def __repr__(self):
        """
        Return a string representation of the vector.

        Returns:
        - str: A string representation of the vector.
        """
        return f"Vector({self.x}, {self.y})"

    def __add__(self, other):
        """
        Define addition for vectors.

        Args:
        - other (Vector): Another vector to add.

        Returns:
        - Vector: The sum of the two vectors.
        """
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        """
        Define subtraction for vectors.

        Args:
        - other (Vector): Another vector to subtract.

        Returns:
        - Vector: The difference between the two vectors.
        """
        return Vector(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        """
        Define scalar multiplication for vectors.

        Args:
        - scalar (float): The scalar value to multiply the vector by.

        Returns:
        - Vector: The result of multiplying the vector by the scalar.
        """
        return Vector(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar):
        """
        Define scalar multiplication when vector is on the right-hand side.

        Args:
        - scalar (float): The scalar value to multiply the vector by.

        Returns:
        - Vector: The result of multiplying the vector by the scalar.
        """
        return self.__mul__(scalar)

# Example: Using Special Methods
# Let's demonstrate the usage of special methods by creating instances of the Vector class.

# Create vector objects
v1 = Vector(1, 2)
v2 = Vector(3, 4)

# Use special methods for addition, subtraction, and scalar multiplication
print("Vector Addition:", v1 + v2)       # Output: Vector(4, 6)
print("Vector Subtraction:", v1 - v2)    # Output: Vector(-2, -2)
print("Scalar Multiplication:", v1 * 2)  # Output: Vector(2, 4)

# Documenting the Vector Class:
def special_methods_documentation():
    """
    This function demonstrates special methods (magic methods) in Python.

    Classes:
    - Vector: A class representing a vector in 2D space.
    """
    pass

# End of example
