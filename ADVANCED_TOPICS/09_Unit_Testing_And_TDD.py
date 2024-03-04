# FileName: 09_Unit_Testing_And_TDD.py
# Code Written by Manuel Cortés Granados
# Date: 2024-03-04
# Location: Bogota DC, Colombia
# LinkedIn Profile: https://www.linkedin.com/in/mcortesgranados/

# Unit Testing and Test-Driven Development (TDD) in Python
# This Python code demonstrates unit testing and Test-Driven Development (TDD) using the unittest module. 
# It includes a Calculator class with an add method to be tested, and a TestCalculator class inheriting from unittest.TestCase containing test methods for the add method. 
# Additionally, it includes metadata such as the proposed filename, authorship information, date and time, location, and a link to the author's LinkedIn profile for context.

import unittest

# Example class with a method to be tested
class Calculator:
    def add(self, x, y):
        return x + y

# Test case class inheriting from unittest.TestCase
class TestCalculator(unittest.TestCase):
    # Test method for the add method of the Calculator class
    def test_add(self):
        calculator = Calculator()
        self.assertEqual(calculator.add(1, 2), 3)
        self.assertEqual(calculator.add(-1, 1), 0)
        self.assertEqual(calculator.add(-1, -1), -2)

if __name__ == "__main__":
    unittest.main()
