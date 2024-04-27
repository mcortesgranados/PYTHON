# Python Class and Instance Variables Example

class MyClass:
    """
    A class demonstrating class variables and instance variables in Python.

    Class Attributes:
    - class_var (int): A class variable shared among all instances.

    Methods:
    - __init__(self, instance_var): Constructor method to initialize an instance of MyClass.
    - instance_method(self): A method to access instance variables.
    """

    class_var = 0   # Class variable shared among all instances

    def __init__(self, instance_var):
        """
        Initialize an instance of MyClass with an instance variable.

        Args:
        - instance_var (int): An instance variable unique to each instance.
        """
        self.instance_var = instance_var

    def instance_method(self):
        """
        A method to access instance variables.
        """
        print("Instance Variable:", self.instance_var)
        print("Class Variable:", MyClass.class_var)

# Example: Using Class and Instance Variables
# Let's demonstrate the usage of class variables and instance variables by creating instances of MyClass.

# Create instances of MyClass with different instance variables
obj1 = MyClass(10)
obj2 = MyClass(20)

# Access instance variables using instance methods
obj1.instance_method()  # Output: Instance Variable: 10, Class Variable: 0
obj2.instance_method()  # Output: Instance Variable: 20, Class Variable: 0

# Access class variable directly
print("Accessing Class Variable Directly:", MyClass.class_var)  # Output: 0

# Modify class variable
MyClass.class_var = 100

# Class variable is shared among all instances
print("Class Variable After Modification:", obj1.class_var)   # Output: 100
print("Class Variable After Modification:", obj2.class_var)   # Output: 100

# Documenting the MyClass Class:
def class_instance_variables_documentation():
    """
    This function demonstrates class variables and instance variables in Python.

    Classes:
    - MyClass: A class demonstrating class variables and instance variables in Python.
    """
    pass

# End of example
