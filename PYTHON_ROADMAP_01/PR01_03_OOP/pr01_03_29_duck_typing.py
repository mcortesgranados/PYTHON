""""

Duck typing is a concept in programming languages like Python, where the type or class of an object is less important than the methods it defines. 
In other words, an object is considered to be of a certain type if it has the necessary methods, regardless of its actual class or inheritance hierarchy. 
Duck typing allows for more flexible and dynamic code, as it focuses on what an object can do rather than what it is. 
Below is an example illustrating duck typing in Python, along with detailed explanations and documentation:

In this example:

- The Duck class represents a duck with quack and fly methods.
- The Airplane class represents an airplane with a fly method.
- The make_it_quack_and_fly function takes an object as input and checks if it has the quack and fly methods using the hasattr function and callable method.
- We pass instances of both Duck and Airplane classes to the make_it_quack_and_fly function, demonstrating that duck typing allows objects of different
 classes to be treated similarly if they have the necessary methods.

This example illustrates how duck typing in Python allows for more flexible and dynamic code by focusing on an object's behavior rather than
 its type or class. As long as an object has the necessary methods, it can be used interchangeably with other objects in functions or methods, 
 promoting code reusability and simplifying interfaces.

"""

# Python Duck Typing Example

class Duck:
    """
    A class representing a duck.
    
    Methods:
    - quack(self): Make the duck quack.
    - fly(self): Make the duck fly.
    """

    def quack(self):
        """
        Make the duck quack.
        """
        return "Quack!"

    def fly(self):
        """
        Make the duck fly.
        """
        return "Flying"


class Airplane:
    """
    A class representing an airplane.
    
    Methods:
    - fly(self): Make the airplane fly.
    """

    def fly(self):
        """
        Make the airplane fly.
        """
        return "Flying"


def make_it_quack_and_fly(obj):
    """
    A function that makes an object quack and fly if it has the necessary methods.
    
    Args:
    - obj (object): An object that is expected to have quack and fly methods.
    """
    if hasattr(obj, 'quack') and callable(obj.quack):
        print("Quack:", obj.quack())

    if hasattr(obj, 'fly') and callable(obj.fly):
        print("Fly:", obj.fly())


# Example: Using Duck Typing
# Let's demonstrate duck typing by passing different objects to the make_it_quack_and_fly function.

# Create a Duck object and make it quack and fly
duck = Duck()
print("Duck:")
make_it_quack_and_fly(duck)  # Output: Quack: Quack! Fly: Flying

# Create an Airplane object and make it fly
airplane = Airplane()
print("\nAirplane:")
make_it_quack_and_fly(airplane)  # Output: Fly: Flying


# Documenting the Duck, Airplane, and make_it_quack_and_fly Function:
def duck_typing_documentation():
    """
    This function demonstrates duck typing in Python.

    Classes:
    - Duck: A class representing a duck with quack and fly methods.
    - Airplane: A class representing an airplane with a fly method.

    Functions:
    - make_it_quack_and_fly(obj): A function that makes an object quack and fly if it has the necessary methods.
    """
    pass

# End of example
