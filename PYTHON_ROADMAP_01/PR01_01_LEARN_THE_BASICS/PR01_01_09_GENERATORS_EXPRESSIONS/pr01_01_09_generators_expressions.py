"""
Generator Expressions in Python (Comprehensive Demonstration)

This script comprehensively demonstrates generator expressions, a memory-efficient
technique for creating iterators on the fly in Python.
"""

# Basic generator expression
numbers = (x for x in range(1, 6))  # Generator of numbers from 1 to 5 (excluding 6)

# Accessing elements (careful, consumes the generator)
print("First number:", next(numbers))  # Use next() to get the first element

try:
  print("Second number:", next(numbers))  # This will raise StopIteration after iterating through all elements
except StopIteration:
  print("Generator is exhausted.")

# Looping through the generator (only iterate once)
for number in numbers:  # Recreates the generator on each loop iteration
  print(number)

# Generator expression with conditionals
even_numbers = (x for x in range(1, 11) if x % 2 == 0)  # Even numbers from 1 to 10

print("\nEven numbers (1 to 10):")
for num in even_numbers:
  print(num)

# Generator expression with modifications
squared_numbers = (x * x for x in range(1, 6))  # Squares from 1 to 5

print("\nSquares (1 to 5):")
for square in squared_numbers:
  print(square)

# Using generator expressions with functions
def is_prime(num):
  """Simple primality check (not optimized for large numbers)"""
  if num <= 1:
    return False
  for i in range(2, int(num**0.5) + 1):
    if num % i == 0:
      return False
  return True

primes = (num for num in range(1, 20) if is_prime(num))  # Prime numbers from 1 to 19

print("\nPrime numbers (1 to 19):")
for prime in primes:
  print(prime)

# Key Points:

* Generator expressions create iterators lazily, yielding elements on demand.
* They are memory-efficient for large datasets or infinite sequences.
* Similar to list comprehensions, they can include conditions and modifications.
* Generator expressions can be used within functions for cleaner code.

"""This script concludes the demonstration of generator expressions."""
