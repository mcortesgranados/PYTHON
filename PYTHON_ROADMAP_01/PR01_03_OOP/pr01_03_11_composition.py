# Python Constructor Composition Example

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
    A class representing a car with constructor composition.

    Attributes:
    - brand (str): The brand of the car.
    - model (str): The model of the car.
    - engine (Engine): The engine of the car.

    Methods:
    - __init__(self, brand, model, engine_type): Constructor method to initialize a car object.
    """

    def __init__(self, brand, model, engine_type):
        """
        Initialize a car with its brand, model, and engine type.

        Args:
        - brand (str): The brand of the car.
        - model (str): The model of the car.
        - engine_type (str): The type of engine for the car.
        """
        self.brand = brand
        self.model = model
        self.engine = Engine(engine_type)  # Constructor composition

# Example: Using Constructor Composition
# Let's demonstrate the usage of constructor composition by creating an instance of the Car class.

# Create a car object with an engine
car = Car("Toyota", "Camry", "V6")

# Display car information
print("Car Brand:", car.brand)        # Output: Toyota
print("Car Model:", car.model)        # Output: Camry
print("Engine Type:", car.engine.type)   # Output: V6

# Documenting the Engine and Car Classes:
def constructor_composition_documentation():
    """
    This function demonstrates constructor composition in Python.

    Classes:
    - Engine: A class representing an engine.
    - Car: A class representing a car with constructor composition.
    """
    pass

# End of example
