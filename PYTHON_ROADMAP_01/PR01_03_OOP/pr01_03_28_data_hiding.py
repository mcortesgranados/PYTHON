"""
Data hiding, also known as encapsulation, is the concept of restricting access to certain attributes or methods of a class. 
This is typically achieved by marking attributes or methods as private, which means they can only be accessed within the class 
itself and not from outside the class. Data hiding helps to enforce information hiding and protects the internal state of an 
object from being modified directly. Below is an example illustrating data hiding in Python, along with detailed explanations and documentation:

In this example:

- The Car class represents a car with private attributes _make, _model, and _year.
- Getter methods (get_make, get_model, get_year) are provided to access the private attributes.
- Setter methods (set_make, set_model, set_year) are provided as private methods to set the private attributes.

Private attributes are marked with a single underscore (_), which is a convention in Python to indicate that they should not be accessed directly 
from outside the class.

This example demonstrates how data hiding in Python can be achieved using private attributes and methods to encapsulate the internal state 
of an object and protect it from direct access and modification from outside the class. Access to private attributes is provided through getter and setter 
methods, promoting encapsulation and information hiding principles.

"""

# Python Data Hiding Example

class Car:
    """
    A class representing a car.
    
    Attributes:
    - _make (str): The make of the car (private).
    - _model (str): The model of the car (private).
    - _year (int): The year of manufacture of the car (private).

    Methods:
    - get_make(self): Get the make of the car.
    - set_make(self, make): Set the make of the car (private).
    - get_model(self): Get the model of the car.
    - set_model(self, model): Set the model of the car (private).
    - get_year(self): Get the year of manufacture of the car.
    - set_year(self, year): Set the year of manufacture of the car (private).
    """

    def __init__(self, make, model, year):
        """
        Initialize a car with its make, model, and year.

        Args:
        - make (str): The make of the car.
        - model (str): The model of the car.
        - year (int): The year of manufacture of the car.
        """
        self._make = make   # Private attribute
        self._model = model   # Private attribute
        self._year = year   # Private attribute

    def get_make(self):
        """
        Get the make of the car.

        Returns:
        - str: The make of the car.
        """
        return self._make

    def set_make(self, make):
        """
        Set the make of the car. This method is private.

        Args:
        - make (str): The make of the car.
        """
        self._make = make

    def get_model(self):
        """
        Get the model of the car.

        Returns:
        - str: The model of the car.
        """
        return self._model

    def set_model(self, model):
        """
        Set the model of the car. This method is private.

        Args:
        - model (str): The model of the car.
        """
        self._model = model

    def get_year(self):
        """
        Get the year of manufacture of the car.

        Returns:
        - int: The year of manufacture of the car.
        """
        return self._year

    def set_year(self, year):
        """
        Set the year of manufacture of the car. This method is private.

        Args:
        - year (int): The year of manufacture of the car.
        """
        self._year = year


# Example: Using Data Hiding
# Let's demonstrate data hiding by creating an instance of the Car class.

# Create a car object
car = Car("Toyota", "Camry", 2020)

# Accessing private attributes directly (which is not recommended)
# This works, but it violates encapsulation and data hiding principles
print("Make (Direct Access):", car._make)  # Output: Toyota

# Accessing private attributes using getter methods
print("Make (Using Getter Method):", car.get_make())  # Output: Toyota

# Attempt to set private attribute directly (which is not allowed)
# This will raise an AttributeError
try:
    car._make = "Honda"   # This should not be allowed
except AttributeError as e:
    print("Error:", e)

# Attempt to set private attribute using setter method (which is allowed)
car.set_make("Honda")
print("Make (After Setting Using Setter Method):", car.get_make())  # Output: Honda

# Documenting the Car Class:
def data_hiding_documentation():
    """
    This function demonstrates data hiding in Python.

    Classes:
    - Car: A class representing a car with private attributes for make, model, and year.
    """
    pass

# End of example
