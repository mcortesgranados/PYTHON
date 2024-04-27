# Python Classes and Objects Example

class Car:
    """
    A class representing a car.

    Attributes:
    - make (str): The make of the car.
    - model (str): The model of the car.
    - year (int): The year of manufacture of the car.
    - color (str): The color of the car.
    - mileage (float): The mileage of the car in kilometers.
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
        self.make = make
        self.model = model
        self.year = year
        self.color = color
        self.mileage = mileage

    def drive(self, distance):
        """
        Simulate driving the car by adding to its mileage.

        Args:
        - distance (float): The distance traveled in kilometers.
        """
        self.mileage += distance

    def __str__(self):
        """
        Return a string representation of the car.

        Returns:
        - str: A string representation of the car.
        """
        return f"{self.year} {self.make} {self.model} ({self.color})"

# Example: Using Classes and Objects
# Let's demonstrate the usage of the Car class by creating car objects and performing operations on them.

# Create car objects
car1 = Car("Toyota", "Camry", 2020, "Red")
car2 = Car("Honda", "Accord", 2018, "Blue", mileage=15000.0)

# Display car information
print("Car 1:", car1)  # Output: 2020 Toyota Camry (Red)
print("Car 2:", car2)  # Output: 2018 Honda Accord (Blue)

# Drive car 1 for 100 kilometers
car1.drive(100)

# Display updated mileage for car 1
print("Car 1 Mileage:", car1.mileage)  # Output: 100.0

# Documenting the Car Class:
def car_documentation():
    """
    This function demonstrates the Car class in Python.

    Car Class:
    - __init__(make, model, year, color, mileage): Initialize a car with its make, model, year, color, and mileage.
    - drive(distance): Simulate driving the car by adding to its mileage.
    - __str__(): Return a string representation of the car.
    """
    pass

# End of example
