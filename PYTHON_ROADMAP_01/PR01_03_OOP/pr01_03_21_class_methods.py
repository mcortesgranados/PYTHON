"""
Class methods in Python are methods that are bound to the class rather than the instances of the class. 
They can access and modify class attributes but not instance attributes. 
Class methods are defined using the @classmethod decorator and have access to the class itself 
via the cls parameter. Below is an example illustrating class methods in Python, 
along with detailed explanations and documentation:
"""

# Python Class Methods Example

class MyClass:
    """
    A class demonstrating class methods in Python.

    Class Attributes:
    - count (int): A class attribute to keep track of the number of instances created.

    Methods:
    - __init__(self, data): Constructor method to initialize an instance of MyClass.
    - get_count(cls): A class method to get the count of instances created.
    """

    count = 0  # Class attribute to keep track of the number of instances created

    def __init__(self, data):
        """
        Initialize an instance of MyClass with data.

        Args:
        - data: Data to be stored in the instance.
        """
        self.data = data
        MyClass.count += 1  # Increment the count of instances created

    @classmethod
    def get_count(cls):
        """
        Get the count of instances created.

        Args:
        - cls: The class itself.

        Returns:
        - int: The count of instances created.
        """
        return cls.count

# Example: Using Class Methods
# Let's demonstrate the usage of class methods by creating instances of MyClass.

# Create instances of MyClass
obj1 = MyClass("Data 1")
obj2 = MyClass("Data 2")
obj3 = MyClass("Data 3")

# Get the count of instances using the class method
print("Count of Instances:", MyClass.get_count())  # Output: 3

# Documenting the MyClass Class:
def class_methods_documentation():
    """
    This function demonstrates class methods in Python.

    Classes:
    - MyClass: A class demonstrating class methods in Python.
    """
    pass

# End of example
