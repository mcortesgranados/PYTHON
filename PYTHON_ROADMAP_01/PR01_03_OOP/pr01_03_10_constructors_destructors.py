# Python Constructors and Destructors Example

class Car:
    """
    A class representing a car with constructors and destructors.

    Attributes:
    - brand (str): The brand of the car.
    - model (str): The model of the car.

    Methods:
    - __init__(self, brand, model): Constructor method to initialize a car object.
    - __del__(self): Destructor method to clean up resources when a car object is destroyed.
    """

    def __init__(self, brand, model):
        """
        Initialize a car with its brand and model.

        Args:
        - brand (str): The brand of the car.
        - model (str): The model of the car.
        """
        self.brand = brand
        self.model = model
        print(f"A {self.brand} {self.model} has been created.")

    def __del__(self):
        """
        Clean up resources when a car object is destroyed.
        """
        print(f"The {self.brand} {self.model} has been destroyed.")

# Example: Using Constructors and Destructors
# Let's demonstrate the usage of constructors and destructors by creating and deleting instances of the Car class.

# Create a car object
car1 = Car("Toyota", "Camry")

# Delete the car object (explicitly)
del car1

# Create another car object
car2 = Car("Honda", "Civic")

# The second car object will be automatically destroyed when the program exits

# Documenting the Car Class:
def constructors_destructors_documentation():
    """
    This function demonstrates constructors and destructors in Python.

    Classes:
    - Car: A class representing a car with constructors and destructors.
    """
    pass

# End of example
