# FileName: 14_C_Extensions_with_Cython.py
# Code Written by Manuel Cortés Granados
# Date: 2024-03-04
# Location: Bogota DC, Colombia
# LinkedIn Profile: https://www.linkedin.com/in/mcortesgranados/

# Working with C Extensions using Cython in Python

# Cython is a superset of the Python language that allows you to write C extensions for Python. 
# It provides a way to easily call C functions and manipulate C data types from Python.

# Example of a Cython module containing C code
# Save this code in a file with .pyx extension, e.g., example.pyx
"""
cdef int c_sum(int a, int b):
    return a + b
"""

# Example of a setup file for building the Cython module
# Save this code in a file named setup.py
"""
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("example.pyx")
)
"""

# Example of how to build and use the Cython module
# Open the terminal and navigate to the directory containing the setup.py file
# Run the command: python setup.py build_ext --inplace

# Now you can import and use the Cython module in your Python code
import example
print("Cython sum:", example.c_sum(3, 4))
