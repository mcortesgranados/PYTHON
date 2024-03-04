# FileName: 23_Working_with_C_Extensions.py
# Code Written by Manuel Cortés Granados
# Date: 2024-03-04
# Location: Bogota DC, Colombia
# LinkedIn Profile: https://www.linkedin.com/in/mcortesgranados/

# Working with C Extensions in Python using Cython

# Cython is a superset of the Python language that allows calling C functions and declaring C types.
# It provides a way to write Python-like code that can be compiled as C extensions, allowing for performance improvements.

# Example: Cython code calling a C function to calculate factorial

# To compile and use Cython code, you need to install the Cython package. Additionally, you may need a C compiler to compile the generated C code.
# Here are the commands to install Cython and a C compiler on different platforms:

# For Linux (Debian/Ubuntu):
# sudo apt-get update
# sudo apt-get install python3 cython3 build-essential

# For macOS (using Homebrew):
# brew install python3 cython

# For Windows (using pip):
# pip install cython

# Make sure you have Python and pip installed on your system before running these commands.
# Additionally, you may need to adjust the commands based on your specific system setup.

# Cython code calling the factorial function
from math import factorial as c_factorial

def calculate_factorial(n):
    return c_factorial(n)

# Test the calculate_factorial function
number = 5
result = calculate_factorial(number)
print(f"Factorial of {number}: {result}")
