import datetime

# Code Written by Manuel Cortés Granados
# Date: 2024-03-04
# Location: Bogota DC, Colombia
# LinkedIn Profile: https://www.linkedin.com/in/mcortesgranados/

# Get current date and time
current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Generators and Iterators in Python

# Generator functions allow you to generate a sequence of values over time, rather than computing them all at once and storing them in memory.
# They are defined using the 'yield' keyword, which suspends the function's execution and yields a value to the caller.

# Define a generator function that yields a sequence of squares
def square_generator(n):
    """
    This generator function yields the squares of numbers from 1 to n.
    """
    for i in range(1, n + 1):
        yield i ** 2

# Create a generator object by calling the generator function
square_gen = square_generator(5)

# Header for the output
print("Demo code of Generators and Iterators in Python")
print("Authored by Manuel Cortés Granados")
print("Date:", current_time)
print("LinkedIn Profile: https://www.linkedin.com/in/mcortesgranados/")
print("-" * 50)  # Separator for a nice display

# Iterate over the generator to access the generated values
print("Squares of numbers from 1 to 5:")
for square in square_gen:
    print(square)

# Output:
# Demo code of Generators and Iterators in Python
# Authored by Manuel Cortés Granados
# Date: 2024-03-04 12:30:45
# LinkedIn Profile: https://www.linkedin.com/in/mcortesgranados/
# --------------------------------------------------
# Squares of numbers from 1 to 5:
# 1
# 4
# 9
# 16
# 25

# Generators are memory-efficient as they produce values on-the-fly, one at a time, rather than storing them all in memory at once.

# Iterator objects allow you to iterate over a sequence of elements.
# They implement the iterator protocol, which requires the __iter__() and __next__() methods.

# Define an iterator class to iterate over a range of numbers
class MyRangeIterator:
    """
    This iterator class iterates over a range of numbers from start to end.
    """
    def __init__(self, start, end):
        self.current = start
        self.end = end

    def __iter__(self):
        return self

    def __next__(self):
        if self.current < self.end:
            value = self.current
            self.current += 1
            return value
        else:
            raise StopIteration

# Create an iterator object by instantiating the iterator class
my_range = MyRangeIterator(1, 5)

# Header for the output
print("\n" + "-" * 50)  # Separator for a nice display
print("Numbers from 1 to 5 using custom iterator:")

# Iterate over the iterator to access the elements
for num in my_range:
    print(num)

# Output:
# --------------------------------------------------
# Numbers from 1 to 5 using custom iterator:
# 1
# 2
# 3
# 4

# Iterators provide a way to traverse sequences of elements without exposing the underlying data structure, thus enhancing encapsulation and abstraction.
