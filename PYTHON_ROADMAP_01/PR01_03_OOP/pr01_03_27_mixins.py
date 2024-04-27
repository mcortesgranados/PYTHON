# Python Mixins Example

# Mixin class for logging functionality
class LoggingMixin:
    """
    A mixin class providing logging functionality.
    
    Methods:
    - log(self, message): Log a message.
    """

    def log(self, message):
        """
        Log a message.

        Args:
        - message (str): The message to be logged.
        """
        print("[LOG] {}: {}".format(type(self).__name__, message))


# Mixin class for serialization functionality
class SerializationMixin:
    """
    A mixin class providing serialization functionality.
    
    Methods:
    - serialize(self): Serialize the object.
    """

    def serialize(self):
        """
        Serialize the object.

        Returns:
        - str: A string representation of the serialized object.
        """
        return str(self.__dict__)


# Example: Using Mixins
# Let's demonstrate the usage of mixins by creating a class that incorporates logging and serialization functionalities.

class MyClass(LoggingMixin, SerializationMixin):
    """
    A class demonstrating the usage of mixins.
    """

    def __init__(self, name, age):
        """
        Initialize MyClass with name and age.

        Args:
        - name (str): The name of the object.
        - age (int): The age of the object.
        """
        self.name = name
        self.age = age


# Create an instance of MyClass
obj = MyClass("John", 30)

# Use logging functionality provided by LoggingMixin
obj.log("Instance created")

# Use serialization functionality provided by SerializationMixin
serialized_data = obj.serialize()
print("Serialized Data:", serialized_data)

# Documenting the LoggingMixin, SerializationMixin, and MyClass classes:
def mixins_documentation():
    """
    This function demonstrates mixins in Python.

    Classes:
    - LoggingMixin: A mixin class providing logging functionality.
    - SerializationMixin: A mixin class providing serialization functionality.
    - MyClass: A class demonstrating the usage of mixins.
    """
    pass

# End of example
