# Python Inheritance Example

class Animal:
    """
    A class representing an animal.

    Attributes:
    - species (str): The species of the animal.
    - legs (int): The number of legs of the animal.
    """

    def __init__(self, species, legs):
        """
        Initialize an animal with its species and number of legs.

        Args:
        - species (str): The species of the animal.
        - legs (int): The number of legs of the animal.
        """
        self.species = species
        self.legs = legs

    def make_sound(self):
        """
        Produce a sound characteristic of the animal.
        """
        pass  # Implementation varies depending on the subclass

class Dog(Animal):
    """
    A class representing a dog, inheriting from the Animal class.

    Attributes:
    - breed (str): The breed of the dog.
    """

    def __init__(self, breed):
        """
        Initialize a dog with its breed.

        Args:
        - breed (str): The breed of the dog.
        """
        super().__init__("Dog", 4)  # Call superclass constructor
        self.breed = breed

    def make_sound(self):
        """
        Produce a bark sound.
        """
        print("Woof!")

class Cat(Animal):
    """
    A class representing a cat, inheriting from the Animal class.

    Attributes:
    - color (str): The color of the cat.
    """

    def __init__(self, color):
        """
        Initialize a cat with its color.

        Args:
        - color (str): The color of the cat.
        """
        super().__init__("Cat", 4)  # Call superclass constructor
        self.color = color

    def make_sound(self):
        """
        Produce a meow sound.
        """
        print("Meow!")

# Example: Using Inheritance
# Let's demonstrate the usage of inheritance by creating instances of the Animal, Dog, and Cat classes.

# Create animal objects
animal = Animal("Unknown", 4)
dog = Dog("Golden Retriever")
cat = Cat("White")

# Display information and make sounds
print("Animal Species:", animal.species)  # Output: Unknown
print("Dog Breed:", dog.breed)            # Output: Golden Retriever
print("Cat Color:", cat.color)            # Output: White
print("Number of Legs (Dog):", dog.legs)  # Output: 4
print("Number of Legs (Cat):", cat.legs)  # Output: 4

# Make sounds
dog.make_sound()  # Output: Woof!
cat.make_sound()  # Output: Meow!

# Documenting the Animal, Dog, and Cat Classes:
def inheritance_documentation():
    """
    This function demonstrates inheritance in Python.

    Classes:
    - Animal: A class representing an animal.
    - Dog: A class representing a dog, inheriting from the Animal class.
    - Cat: A class representing a cat, inheriting from the Animal class.
    """
    pass

# End of example
