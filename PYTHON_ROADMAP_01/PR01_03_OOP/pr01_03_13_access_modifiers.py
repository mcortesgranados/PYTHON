"""
Access modifiers are keywords in object-oriented programming languages that specify the accessibility of class 
members (attributes and methods) from outside the class. In Python, access modifiers are not strictly enforced 
like in some other languages, but conventions are used to indicate the intended accessibility. The common conventions are:

Public: Attributes and methods are accessible from outside the class.
Protected: Attributes and methods are accessible only within the class and its subclasses.
Private: Attributes and methods are accessible only within the class itself.

"""

# Python Access Modifiers Example

class MyClass:
    """
    A class demonstrating access modifiers in Python.

    Attributes:
    - public_attr (int): A public attribute.
    - _protected_attr (int): A protected attribute.
    - __private_attr (int): A private attribute.

    Methods:
    - public_method(self): A public method.
    - _protected_method(self): A protected method.
    - __private_method(self): A private method.
    """

    def __init__(self):
        """
        Initialize an instance of MyClass with default values.
        """
        self.public_attr = 10            # Public attribute
        self._protected_attr = 20        # Protected attribute
        self.__private_attr = 30         # Private attribute

    def public_method(self):
        """
        A public method that accesses all attributes.
        """
        print("Public Method:")
        print("Public Attribute:", self.public_attr)
        print("Protected Attribute:", self._protected_attr)
        print("Private Attribute:", self.__private_attr)
        self.__private_method()

    def _protected_method(self):
        """
        A protected method that accesses all attributes.
        """
        print("Protected Method:")
        print("Public Attribute:", self.public_attr)
        print("Protected Attribute:", self._protected_attr)
        print("Private Attribute:", self.__private_attr)
        self.__private_method()

    def __private_method(self):
        """
        A private method that accesses all attributes.
        """
        print("Private Method:")
        print("Public Attribute:", self.public_attr)
        print("Protected Attribute:", self._protected_attr)
        print("Private Attribute:", self.__private_attr)

# Example: Using Access Modifiers
# Let's demonstrate the usage of access modifiers by creating an instance of MyClass and accessing its attributes and methods.

# Create an instance of MyClass
my_obj = MyClass()

# Access public attribute and method
print("Accessing Public Attribute:", my_obj.public_attr)    # Output: 10
my_obj.public_method()

# Access protected attribute and method (conventionally allowed, but discouraged)
print("Accessing Protected Attribute:", my_obj._protected_attr)  # Output: 20
my_obj._protected_method()

# Accessing private attribute and method (conventionally discouraged)
# The following lines will raise AttributeError because private attributes and methods are not accessible from outside the class
# print("Accessing Private Attribute:", my_obj.__private_attr)
# my_obj.__private_method()

# Documenting the MyClass Class:
def access_modifiers_documentation():
    """
    This function demonstrates access modifiers in Python.

    Classes:
    - MyClass: A class demonstrating access modifiers in Python.
    """
    pass

# End of example
