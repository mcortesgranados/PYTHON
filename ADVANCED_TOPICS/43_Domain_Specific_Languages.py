# FileName: 43_Domain_Specific_Languages.py
# Code Written by Manuel Cortés Granados
# Date: 2024-03-04
# Location: Bogota DC, Colombia
# LinkedIn Profile: https://www.linkedin.com/in/mcortesgranados/

# Creating Domain-Specific Languages (DSLs) with Python

# Python's flexibility and expressive syntax make it suitable for creating domain-specific languages (DSLs) tailored to specific problem domains.

class Calculator:
    def __init__(self):
        self._result = 0

    def add(self, num):
        self._result += num

    def subtract(self, num):
        self._result -= num

    def multiply(self, num):
        self._result *= num

    def divide(self, num):
        self._result /= num

    def result(self):
        return self._result

# Example: Creating a simple DSL for arithmetic operations
calc = Calculator()
calc.add(5)
calc.subtract(3)
calc.multiply(2)
calc.divide(4)

print("Result of arithmetic operations:", calc.result())
