"""

Hierarchical inheritance in Python refers to the process of multiple subclasses inheriting from a single superclass. 
This creates a hierarchy of classes where each subclass shares common attributes and methods inherited from the superclass. 
Below is an example illustrating hierarchical inheritance in Python, along with detailed explanations and documentation:

"""

# Python Hierarchical Inheritance Example

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
    - speak(self): Make the dog bark.
    """

    def speak(self):
        """
        Make the dog bark.

        Returns:
        - str: A string representing the bark of the dog.
        """
        return "Woof!"


class Cat(Animal):
    """
    A subclass representing a cat, inheriting from the Animal class.
    
    Methods:
    - speak(self): Make the cat meow.
    """

    def speak(self):
        """
        Make the cat meow.

        Returns:
        - str: A string representing the meow of the cat.
        """
        return "Meow!"


# Example: Using Hierarchical Inheritance
# Let's demonstrate hierarchical inheritance by creating instances of the Dog and Cat classes.

# Create a dog object
dog = Dog("Canine")

# Access the species attribute inherited from the Animal class
print("Dog Species:", dog.species)  # Output: Canine

# Make the dog speak using the inherited speak method
print("Dog Speak:", dog.speak())    # Output: Woof!


# Create a cat object
cat = Cat("Feline")

# Access the species attribute inherited from the Animal class
print("Cat Species:", cat.species)  # Output: Feline

# Make the cat speak using the inherited speak method
print("Cat Speak:", cat.speak())    # Output: Meow!


# Documenting the Animal, Dog, and Cat Classes:
def hierarchical_inheritance_documentation():
    """
    This function demonstrates hierarchical inheritance in Python.

    Classes:
    - Animal: A base class representing an animal.
    - Dog: A subclass representing a dog, inheriting from the Animal class.
    - Cat: A subclass representing a cat, inheriting from the Animal class.
    """
    pass

# End of example
