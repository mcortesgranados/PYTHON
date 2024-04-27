"""
Class attributes in Python are attributes that are associated with the class itself rather than with instances of the class. 
They are shared among all instances of the class and can be accessed using the class name or any instance of the class. 
Below is an example illustrating class attributes in Python, along with detailed explanations and documentation:

"""
# Python Class Attributes Example

class Dog:
    """
    A class representing a dog with class attributes.

    Class Attributes:
    - species (str): The species of the dog.
    - legs (int): The number of legs a dog has.

    Methods:
    - __init__(self, name): Constructor method to initialize a dog object.
    - describe(self): Describe the dog.
    """

    species = "Canine"  # Class attribute shared among all instances
    legs = 4            # Class attribute shared among all instances

    def __init__(self, name):
        """
        Initialize a dog with its name.

        Args:
        - name (str): The name of the dog.
        """
        self.name = name

    def describe(self):
        """
        Describe the dog.

        Returns:
        - str: A description of the dog.
        """
        return f"{self.name} is a {self.species} with {self.legs} legs."

# Example: Using Class Attributes
# Let's demonstrate the usage of class attributes by creating instances of the Dog class.

# Create dog objects
dog1 = Dog("Buddy")
dog2 = Dog("Max")

# Access class attributes using class name or instance
print("Dog 1 Species:", Dog.species)       # Output: Canine
print("Dog 2 Legs:", dog2.legs)            # Output: 4

# Access instance attribute
print("Dog 1 Name:", dog1.name)            # Output: Buddy

# Describe dogs using instance method
print(dog1.describe())  # Output: Buddy is a Canine with 4 legs.
print(dog2.describe())  # Output: Max is a Canine with 4 legs.

# Documenting the Dog Class:
def class_attributes_documentation():
    """
    This function demonstrates class attributes in Python.

    Classes:
    - Dog: A class representing a dog with class attributes.
    """
    pass

# End of example
