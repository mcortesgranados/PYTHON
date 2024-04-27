"""

Multiple inheritance in Python refers to the process of a class inheriting attributes and methods from multiple parent classes. 
This means that a subclass can inherit from more than one superclass. Below is an example illustrating multiple inheritance in Python, 
along with detailed explanations and documentation:

"""

# Python Multiple Inheritance Example

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


class Pet:
    """
    A base class representing a pet.
    
    Attributes:
    - name (str): The name of the pet.

    Methods:
    - play(self): Abstract method to make the pet play.
    """

    def __init__(self, name):
        """
        Initialize a pet with its name.

        Args:
        - name (str): The name of the pet.
        """
        self.name = name

    def play(self):
        """
        Make the pet play. This is an abstract method and should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement the play() method.")


class Dog(Animal, Pet):
    """
    A subclass representing a dog, inheriting from both Animal and Pet classes.
    
    Methods:
    - speak(self): Make the dog speak.
    - play(self): Make the dog play.
    """

    def __init__(self, name):
        """
        Initialize a dog with its name.

        Args:
        - name (str): The name of the dog.
        """
        Animal.__init__(self, "Canine")
        Pet.__init__(self, name)

    def speak(self):
        """
        Make the dog bark.

        Returns:
        - str: A string representing the bark of the dog.
        """
        return "Woof!"


# Example: Using Multiple Inheritance
# Let's demonstrate multiple inheritance by creating an instance of the Dog class.

# Create a dog object
dog = Dog("Buddy")

# Access the species attribute inherited from the Animal class
print("Dog Species:", dog.species)  # Output: Canine

# Access the name attribute inherited from the Pet class
print("Dog Name:", dog.name)        # Output: Buddy

# Make the dog speak using the inherited speak method
print("Dog Speak:", dog.speak())    # Output: Woof!

# Make the dog play using the inherited play method
dog.play()  # Output: NotImplementedError: Subclasses must implement the play() method.


# Documenting the Animal, Pet, and Dog Classes:
def multiple_inheritance_documentation():
    """
    This function demonstrates multiple inheritance in Python.

    Classes:
    - Animal: A base class representing an animal.
    - Pet: A base class representing a pet.
    - Dog: A subclass representing a dog, inheriting from both Animal and Pet classes.
    """
    pass

# End of example
