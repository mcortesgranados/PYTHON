"""

Single inheritance in Python refers to the process of a class inheriting attributes and methods from a single parent class. 
This means that a subclass can inherit from only one superclass. Below is an example illustrating single inheritance in 
Python, along with detailed explanations and documentation:

"""

# Python Single Inheritance Example

class Animal:
    """
    A base class representing an animal.
    
    Attributes:
    - species (str): The species of the animal.

    Methods:
    - speak(self): Abstract method to make the animal speak.
    """

    def __init__(self, species):
        """
        Initialize an animal with its species.

        Args:
        - species (str): The species of the animal.
        """
        self.species = species

    def speak(self):
        """
        Make the animal speak. This is an abstract method and should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement the speak() method.")


class Dog(Animal):
    """
    A subclass representing a dog, inheriting from the Animal class.
    
    Methods:
    - speak(self): Make the dog speak.
    """

    def speak(self):
        """
        Make the dog bark.

        Returns:
        - str: A string representing the bark of the dog.
        """
        return "Woof!"


# Example: Using Single Inheritance
# Let's demonstrate single inheritance by creating an instance of the Dog class.

# Create a dog object
dog = Dog("Canine")

# Access the species attribute inherited from the Animal class
print("Dog Species:", dog.species)  # Output: Canine

# Make the dog speak using the inherited speak method
print("Dog Speak:", dog.speak())    # Output: Woof!


# Documenting the Animal and Dog Classes:
def single_inheritance_documentation():
    """
    This function demonstrates single inheritance in Python.

    Classes:
    - Animal: A base class re
