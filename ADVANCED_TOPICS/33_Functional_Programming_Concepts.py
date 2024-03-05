# FileName: 33_Functional_Programming_Concepts.py
# Code Written by Manuel Cortés Granados
# Date: 2024-03-04
# Location: Bogota DC, Colombia
# LinkedIn Profile: https://www.linkedin.com/in/mcortesgranados/

# Functional Programming Concepts in Python

# Python supports functional programming paradigms such as higher-order functions and closures.

# Example 1: Higher-order function

# Function that takes another function as an argument
def apply_operation(func, x, y):
    return func(x, y)

# Function to add two numbers
def add(x, y):
    return x + y

# Function to subtract two numbers
def subtract(x, y):
    return x - y

# Call apply_operation with different operations
print("Addition result:", apply_operation(add, 5, 3))
print("Subtraction result:", apply_operation(subtract, 5, 3))

# Example 2: Closure

# Function that returns another function
def outer_function(x):
    def inner_function(y):
        return x + y
    return inner_function

# Create closure functions with different values of 'x'
closure1 = outer_function(5)
closure2 = outer_function(10)

# Call the closure functions
print("Closure 1 result:", closure1(3))
print("Closure 2 result:", closure2(3))
