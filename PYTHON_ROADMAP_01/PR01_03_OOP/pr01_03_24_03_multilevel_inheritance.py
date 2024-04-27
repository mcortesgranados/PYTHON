"""

In this example:

The Animal class is a base class representing an animal, and the Pet class is a subclass representing a pet, inheriting from Animal.
The Dog class is a subclass of Pet, inheriting attributes and methods from both Animal and Pet classes.
When we create an instance of the Dog class, it inherits attributes from both Animal and Pet classes.
This example illustrates how multilevel inheritance in Python allows a subclass to inherit attributes and methods from both its immediate 
superclass and its superclass's superclass. Multilevel inheritance can create a hierarchical class structure, 
but it may also lead to increased complexity and potential issues with code maintenance.

Multilevel inheritance in Python refers to the process of a subclass inheriting attributes and methods from another subclass, 
which in turn inherits from another superclass. This creates a hierarchy of classes where each subclass inherits from its 
immediate superclass, and so on. Below is an example illustrating multilevel inheritance in Python, 
along with detailed explanations and documentation:

"""

# Python Multilevel Inheritance Example

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


class Pet(Animal):
    """
    A subclass representing a pet, inheriting from the Animal class.
    
    Attributes:
    - name (str): The name of the pet.

    Methods:
    - play(self): Make the pet play.
    """

    def __init__(self, species, name):
        """
        Initialize a pet with its species and name.

        Args:
        - species (str): The species of the pet.
        - name (str): The name of the pet.
        """
        super().__init__(species)
        self.name = name

    def play(self):
        """
        Make the pet play. This is an abstract method and should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement the play() method.")


class Dog(Pet):
    """
    A subclass representing a dog, inheriting from the Pet class.
    
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


# Example: Using Multilevel Inheritance
# Let's demonstrate multilevel inheritance by creating an instance of the Dog class.

# Create a dog object
dog = Dog("Canine", "Buddy")

# Access the species attribute inherited from the Animal class
print("Dog Species:", dog.species)  # Output: Canine

# Access the name attribute inherited from the Pet class
print("Dog Name:", dog.name)        # Output: Buddy

# Make the dog speak using the inherited speak method
print("Dog Speak:", dog.speak())    # Output: Woof!

# Make the dog play using the inherited play method
dog.play()  # Output: NotImplementedError: Subclasses must implement the play() method.


# Documenting the Animal, Pet, and Dog Classes:
def multilevel_inheritance_documentation():
    """
    This function demonstrates multilevel inheritance in Python.

    Classes:
    - Animal: A base class representing an animal.
    - Pet: A subclass representing a pet, inheriting from the Animal class.
    - Dog: A subclass representing a dog, inheriting from the Pet class.
    """
    pass

# End of example
