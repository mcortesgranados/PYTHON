"""
 In Python, MRO stands for Method Resolution Order, which defines the order in which methods are searched for and invoked in
  the inheritance hierarchy. Understanding MRO is crucial for resolving method conflicts and ensuring the correct method is called 
  when dealing with multiple inheritance. Below is an example illustrating MRO in Python, along with detailed explanations and documentation:
"""
# Python Method Resolution Order (MRO) Example

class A:
    """
    A base class representing class A.
    """

    def greet(self):
        """
        A method to greet from class A.
        """
        return "Hello from A"

class B(A):
    """
    A class representing class B, inheriting from class A.
    """

    def greet(self):
        """
        A method to greet from class B.
        """
        return "Hello from B"

class C(A):
    """
    A class representing class C, inheriting from class A.
    """

    def greet(self):
        """
        A method to greet from class C.
        """
        return "Hello from C"

class D(B, C):
    """
    A class representing class D, inheriting from class B and class C.
    """

    def greet(self):
        """
        A method to greet from class D.
        """
        return "Hello from D"

# Example: Using Method Resolution Order (MRO)
# Let's demonstrate the method resolution order by creating an instance of class D.

# Create an instance of class D
obj_d = D()

# Call the greet method
print("Greeting:", obj_d.greet())  # Output: Hello from B

# Display the method resolution order
print("Method Resolution Order (MRO):", D.mro())

# Documenting the classes:
def mro_documentation():
    """
    This function demonstrates Method Resolution Order (MRO) in Python.

    Classes:
    - A: A base class representing class A.
    - B: A class representing class B, inheriting from class A.
    - C: A class representing class C, inheriting from class A.
    - D: A class representing class D, inheriting from class B and class C.
    """
    pass

# End of example
