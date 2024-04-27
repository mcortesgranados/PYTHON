# Python Method Overriding Example

class Animal:
    """
    A base class representing an animal.

    Methods:
    - make_sound(): Produce a generic sound characteristic of the animal.
    """

    def make_sound(self):
        """
        Produce a generic sound characteristic of the animal.
        """
        return "Generic animal sound"

class Dog(Animal):
    """
    A class representing a dog, inheriting from the Animal class.

    Methods:
    - make_sound(): Produce a bark sound specific to dogs.
    """

    def make_sound(self):
        """
        Produce a bark sound specific to dogs.
        """
        return "Woof!"

class Cat(Animal):
    """
    A class representing a cat, inheriting from the Animal class.

    Methods:
    - make_sound(): Produce a meow sound specific to cats.
    """

    def make_sound(self):
        """
        Produce a meow sound specific to cats.
        """
        return "Meow!"

# Example: Using Method Overriding
# Let's demonstrate the usage of method overriding by creating instances of the Dog and Cat classes.

# Create animal objects
dog = Dog()
cat = Cat()

# Make sounds using overridden methods
print("Dog Sound:", dog.make_sound())  # Output: Woof!
print("Cat Sound:", cat.make_sound())  # Output: Meow!

# Documenting the Animal, Dog, and Cat Classes:
def method_overriding_documentation():
    """
    This function demonstrates method overriding in Python.

    Classes:
    - Animal: A base class representing an animal.
    - Dog: A class representing a dog, inheriting from the Animal class.
    - Cat: A class representing a cat, inheriting from the Animal class.
    """
    pass

# End of example
