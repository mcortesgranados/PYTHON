"""
Factory methods are a design pattern in object-oriented programming that provide a way to create objects without exposing 
the instantiation logic to the client. Instead of using constructors to create objects directly, clients call a factory method, 
which is responsible for creating instances of objects. This allows for more flexible object creation and can encapsulate complex 
instantiation logic. Below is an example illustrating factory methods in Python, along with detailed explanations and documentation:

In this example:

- The Dog class represents a dog with a name attribute and a speak method.
- The Cat class represents a cat with a name attribute and a speak method.
- The PetFactory class is a factory class with a static method create_pet that creates instances of pets based on the species specified (dog or cat).
- We use the factory method create_pet to create instances of Dog and Cat classes without exposing the instantiation logic to the client.

This example illustrates how factory methods in Python provide a flexible and encapsulated way to create objects, allowing clients to 
create instances of objects without being concerned about the details of object creation. Factory methods can encapsulate complex instantiation
logic and promote code reusability and maintainability.

"""

# Python Factory Method Example

class Dog:
    """
    A class representing a dog.
    
    Attributes:
    - name (str): The name of the dog.

    Methods:
    - speak(self): Make the dog bark.
    """

    def __init__(self, name):
        """
        Initialize a dog with its name.

        Args:
        - name (str): The name of the dog.
        """
        self.name = name

    def speak(self):
        """
        Make the dog bark.

        Returns:
        - str: A string representing the bark of the dog.
        """
        return "Woof!"


class Cat:
    """
    A class representing a cat.
    
    Attributes:
    - name (str): The name of the cat.

    Methods:
    - speak(self): Make the cat meow.
    """

    def __init__(self, name):
        """
        Initialize a cat with its name.

        Args:
        - name (str): The name of the cat.
        """
        self.name = name

    def speak(self):
        """
        Make the cat meow.

        Returns:
        - str: A string representing the meow of the cat.
        """
        return "Meow!"


class PetFactory:
    """
    A factory class responsible for creating instances of pets.
    """

    @staticmethod
    def create_pet(species, name):
        """
        Create a pet object based on the species.

        Args:
        - species (str): The species of the pet ('dog' or 'cat').
        - name (str): The name of the pet.

        Returns:
        - object: An instance of the specified pet species.
        """
        if species.lower() == 'dog':
            return Dog(name)
        elif species.lower() == 'cat':
            return Cat(name)
        else:
            raise ValueError("Invalid pet species: {}".format(species))


# Example: Using Factory Method
# Let's demonstrate the usage of the factory method by creating instances of pets.

# Create a dog using the factory method
dog = PetFactory.create_pet('dog', 'Buddy')
print("Dog:", dog.name)       # Output: Buddy
print("Dog Speak:", dog.speak())  # Output: Woof!

# Create a cat using the factory method
cat = PetFactory.create_pet('cat', 'Whiskers')
print("\nCat:", cat.name)       # Output: Whiskers
print("Cat Speak:", cat.speak())  # Output: Meow!


# Documenting the Dog, Cat, and PetFactory Classes:
def factory_method_documentation():
    """
    This function demonstrates factory methods in Python.

    Classes:
    - Dog: A class representing a dog with a name and a speak method.
    - Cat: A class representing a cat with a name and a speak method.
    - PetFactory: A factory class responsible for creating instances of pets using a factory method.
    """
    pass

# End of example
