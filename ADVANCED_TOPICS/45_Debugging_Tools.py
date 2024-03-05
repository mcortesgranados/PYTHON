# FileName: 45_Debugging_Tools.py
# Code Written by Manuel Cortés Granados
# Date: 2024-03-04
# Location: Bogota DC, Colombia
# LinkedIn Profile: https://www.linkedin.com/in/mcortesgranados/

# Debugging Tools

# Python provides the pdb module for interactive debugging, allowing you to step through code, inspect variables, and diagnose issues.

import pdb

# Example: Debugging a function to calculate factorial
def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

# Call the function with a breakpoint
pdb.set_trace()
print("Factorial of 5:", factorial(5))
