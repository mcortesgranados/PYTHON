# FileName: 34_Memory_Management.py
# Code Written by Manuel Cortés Granados
# Date: 2024-03-04
# Location: Bogota DC, Colombia
# LinkedIn Profile: https://www.linkedin.com/in/mcortesgranados/

# Memory Management in Python

# Python automatically manages memory through garbage collection and provides tools for dealing with weak references.

import gc
import weakref

# Example 1: Garbage Collection

# Create a class with a destructor method
class MyClass:
    def __del__(self):
        print("Object destroyed")

# Create an object
obj = MyClass()

# Delete the object explicitly
del obj

# Example 2: Weak References

# Create an object
my_dict = {'key': 'value'}

# Create a weak reference to the object
weak_ref = weakref.ref(my_dict)

# Get the object from the weak reference
retrieved_obj = weak_ref()

# Print the retrieved object
print("Retrieved Object:", retrieved_obj)

# Force garbage collection
gc.collect()
