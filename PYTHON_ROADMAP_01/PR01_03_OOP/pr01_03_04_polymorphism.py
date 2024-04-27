# Python Polymorphism Example

class Animal:
    """
    A class representing an animal.

    Methods:
    - make_sound(): Produce a sound characteristic of the animal.
    """

    def make_sound(self):
        """
        Produce a sound characteristic of the animal.
        """
        pass  # Implementation varies depending on the subclass

class Dog(Animal):
    """
    A class representing a dog, inheriting from the Animal class.

    Methods:
    - make_sound(): Produce a bark sound.
    """

    def make_sound(self):
        """
        Produce a bark sound.
        """
        return "Woof!"

class Cat(Animal):
    """
    A class representing a cat, inheriting from the Animal class.

    Methods:
    - make_sound(): Produce a meow sound.
    """

    def make_sound(self):
        """
        Produce a meow sound.
        """
        return "Meow!"

# Example: Using Polymorphism
# Let's demonstrate the usage of polymorphism by creating instances of the Dog and Cat classes.

# Create animal objects
dog = Dog()
cat = Cat()

# Make sounds using the same interface
print("Dog Sound:", dog.make_sound())  # Output: Woof!
print("Cat Sound:", cat.make_sound())  # Output: Meow!

# Documenting the Animal, Dog, and Cat Classes:
def polymorphism_documentation():
    """
    This function demonstrates polymorphism in Python.

    Classes:
    - Animal: A class representing an animal.
    - Dog: A class representing a dog, inheriting from the Animal class.
    - Cat: A class representing a cat, inheriting from the Animal class.
    """
    pass

# End of example
