"""
Aggregation is a form of association where one class (the container) contains references to other classes (the contained objects)
 as part of its state. It represents a "has-a" relationship, where the container class has instances of another class as its attributes. B
"""
# Python Aggregation Example

class Engine:
    """
    A class representing an engine.

    Attributes:
    - type (str): The type of engine (e.g., "V6", "V8").
    """

    def __init__(self, engine_type):
        """
        Initialize an engine with its type.

        Args:
        - engine_type (str): The type of engine.
        """
        self.type = engine_type

class Car:
    """
    A class representing a car with aggregation.

    Attributes:
    - brand (str): The brand of the car.
    - model (str): The model of the car.
    - engine (Engine): The engine of the car.

    Methods:
    - __init__(self, brand, model, engine): Constructor method to initialize a car object.
    """

    def __init__(self, brand, model, engine):
        """
        Initialize a car with its brand, model, and engine.

        Args:
        - brand (str): The brand of the car.
        - model (str): The model of the car.
        - engine (Engine): The engine object associated with the car.
        """
        self.brand = brand
        self.model = model
        self.engine = engine  # Aggregation: Car "has-a" Engine

# Example: Using Aggregation
# Let's demonstrate the usage of aggregation by creating instances of the Engine and Car classes.

# Create an engine object
engine = Engine("V6")

# Create a car object with the engine
car = Car("Toyota", "Camry", engine)

# Display car information including engine type
print("Car Brand:", car.brand)            # Output: Toyota
print("Car Model:", car.model)            # Output: Camry
print("Engine Type:", car.engine.type)       # Output: V6

# Documenting the Engine and Car Classes:
def aggregation_documentation():
    """
    This function demonstrates aggregation in Python.

    Classes:
    - Engine: A class representing an engine.
    - Car: A class representing a car with aggregation.
    """
    pass

# End of example
