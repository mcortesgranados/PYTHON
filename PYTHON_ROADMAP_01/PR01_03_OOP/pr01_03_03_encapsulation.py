# Python Encapsulation Example

class Car:
    """
    A class representing a car with encapsulation.

    Attributes:
    - _make (str): The make of the car.
    - _model (str): The model of the car.
    - _year (int): The year of manufacture of the car.
    - _color (str): The color of the car.
    - _mileage (float): The mileage of the car in kilometers.
    """

    def __init__(self, make, model, year, color, mileage=0.0):
        """
        Initialize a car with its make, model, year, color, and optionally mileage.

        Args:
        - make (str): The make of the car.
        - model (str): The model of the car.
        - year (int): The year of manufacture of the car.
        - color (str): The color of the car.
        - mileage (float, optional): The mileage of the car in kilometers. Default is 0.0.
        """
        self._make = make
        self._model = model
        self._year = year
        self._color = color
        self._mileage = mileage

    def get_make(self):
        """
        Get the make of the car.

        Returns:
        - str: The make of the car.
        """
        return self._make

    def set_make(self, make):
        """
        Set the make of the car.

        Args:
        - make (str): The make of the car.
        """
        self._make = make

    def drive(self, distance):
        """
        Simulate driving the car by adding to its mileage.

        Args:
        - distance (float): The distance traveled in kilometers.
        """
        self._mileage += distance

    def get_mileage(self):
        """
        Get the mileage of the car.

        Returns:
        - float: The mileage of the car in kilometers.
        """
        return self._mileage

# Example: Using Encapsulation
# Let's demonstrate the usage of encapsulation by creating a Car object and performing operations on it.

# Create a car object
car = Car("Toyota", "Camry", 2020, "Red")

# Display car information using encapsulated attributes and methods
print("Car Make:", car.get_make())        # Output: Toyota
print("Initial Mileage:", car.get_mileage())  # Output: 0.0

# Drive the car for 100 kilometers
car.drive(100)

# Display updated mileage
print("Updated Mileage:", car.get_mileage())  # Output: 100.0

# Documenting the Car Class:
def encapsulation_documentation():
    """
    This function demonstrates encapsulation in Python.

    Car Class:
    - __init__(make, model, year, color, mileage): Initialize a car with its make, model, year, color, and mileage.
    - get_make(): Get the make of the car.
    - set_make(make): Set the make of the car.
    - drive(distance): Simulate driving the car by adding to its mileage.
    - get_mileage(): Get the mileage of the car.
    """
    pass

# End of example
