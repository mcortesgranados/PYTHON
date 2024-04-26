"""
List Comprehensions in Python (Professional Demonstration)

This script comprehensively demonstrates cd pr01list comprehensions, a powerful
technique for concise and efficient list creation in Python.
"""

# Basic list comprehension
numbers = [x for x in range(1, 6)]  # List of numbers from 1 to 5 (excluding 6)
print("List of numbers (1 to 5):", numbers)

# Conditional list comprehension
even_numbers = [x for x in range(1, 11) if x % 2 == 0]  # Even numbers from 1 to 10
print("List of even numbers (1 to 10):", even_numbers)

# List comprehension with modifications
squared_numbers = [x * x for x in range(1, 6)]  # Squares from 1 to 5
print("List of squares (1 to 5):", squared_numbers)

# Nested list comprehensions
cartesian_product = [(x, y) for x in range(1, 4) for y in range(1, 4)]  # All combinations of (x, y) from 1 to 3
print("Cartesian product (combinations of 1 to 3):", cartesian_product)

# List comprehension with string manipulation
uppercased_fruits = [fruit.upper() for fruit in ["apple", "banana", "cherry"]]  # Uppercase each fruit name
print("Uppercased fruits:", uppercased_fruits)

# List comprehension with filtering based on length
long_words = [word for word in ["hello", "world", "programming"] if len(word) > 5]  # Words longer than 5 characters
print("Words longer than 5 characters:", long_words)

# Using list comprehension with other iterables
letters = [letter for letter in "python"]  # List of letters from the string "python"
print("List of letters from 'python':", letters)

# List comprehension with dictionary creation
fruit_prices = {"apple": 1.50, "banana": 0.75, "orange": 1.25}
expensive_fruits = {fruit for fruit, price in fruit_prices.items() if price > 1.0}  # Set of expensive fruits (price > $1)
print("Set of expensive fruits (price > $1):", expensive_fruits)

# Key Points:
#
#* List comprehensions offer a concise and readable alternative to traditional for loops for list creation.
#* They can be used with various iterables like strings, ranges, and dictionaries.
#* Conditional expressions allow filtering elements within the comprehension itself.
#* Nested comprehensions enable creating complex data structures like combinations.
#* String manipulation and custom logic can be incorporated for versatile use cases.

"""This script concludes the demonstration of list comprehensions."""
