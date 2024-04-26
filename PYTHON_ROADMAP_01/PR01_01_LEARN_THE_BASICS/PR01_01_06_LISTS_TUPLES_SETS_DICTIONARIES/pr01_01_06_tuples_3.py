# Python Tuples Example

# Example 1: Creating a Tuple
# Tuples in Python are ordered collections of items enclosed within parentheses ().
# They can contain any data type (integers, strings, floats, other tuples, etc.).
# Here's an example of creating a tuple:
my_tuple = (1, 2, 3, 4, 5)

# Example 2: Accessing Elements in a Tuple
# You can access individual elements in a tuple using indexing.
# Indexing starts from 0 for the first element, 1 for the second, and so on.
# Here's how you can access elements in the tuple:
first_element = my_tuple[0]
second_element = my_tuple[1]
print("First Element:", first_element)
print("Second Element:", second_element)

# Example 3: Slicing Tuples
# You can slice tuples to extract a subset of elements using a start and end index.
# Slicing returns a new tuple containing the specified range of elements.
# Here's how you can slice a tuple:
subset = my_tuple[1:4]  # Extract elements from index 1 to index 3 (end index is exclusive)
print("Subset:", subset)

# Example 4: Immutable Nature of Tuples
# Tuples are immutable, meaning their elements cannot be changed after they are created.
# However, you can create new tuples by combining or modifying existing tuples.
# Here's an example illustrating the immutable nature of tuples:
try:
    my_tuple[2] = 10  # Trying to modify an element in the tuple (raises TypeError)
except TypeError as e:
    print("Error:", e)

# Example 5: Tuple Methods
# Python provides a few built-in methods to manipulate tuples.
# Here are some commonly used tuple methods:
# - count(): Returns the number of occurrences of a value in the tuple.
# - index(): Returns the index of the first occurrence of a value in the tuple.
# Here's how you can use these methods:
numbers = (1, 2, 3, 4, 5, 1, 3, 4)
count = numbers.count(3)   # Count the number of occurrences of value 3
index = numbers.index(4)   # Find the index of the first occurrence of value 4
print("Count of 3:", count)
print("Index of 4:", index)

# Example 6: Nested Tuples
# You can have tuples within a tuple, known as nested tuples.
# Here's an example of a nested tuple:
nested_tuple = ((1, 2), (3, 4), (5, 6))

# Example 7: Tuple Packing and Unpacking
# Tuple packing is the process of packing multiple values into a single tuple.
# Tuple unpacking is the process of extracting values from a tuple into individual variables.
# Here's how you can perform tuple packing and unpacking:
packed_tuple = 1, 2, 3   # Packing multiple values into a single tuple
a, b, c = packed_tuple   # Unpacking values from a tuple into individual variables
print("Packed Tuple:", packed_tuple)
print("Unpacked Variables:", a, b, c)

# Example 8: Tuple Comprehension (Generator Expressions)
# While there's no direct tuple comprehension in Python like list comprehension,
# you can use generator expressions to create tuples.
# Here's an example of using a generator expression to create a tuple:
squares_tuple = tuple(x**2 for x in range(1, 6))
print("Squares Tuple:", squares_tuple)

# Documenting the Tuples:
def tuples_documentation():
    """
    This function demonstrates various aspects of tuples in Python.

    Example 1:
    - Creating a Tuple: How to define a tuple in Python.

    Example 2:
    - Accessing Elements in a Tuple: How to access elements using indexing.

    Example 3:
    - Slicing Tuples: How to extract a subset of elements using slicing.

    Example 4:
    - Immutable Nature of Tuples: How tuples are immutable.

    Example 5:
    - Tuple Methods: Commonly used methods to manipulate tuples.

    Example 6:
    - Nested Tuples: How to create tuples within tuples.

    Example 7:
    - Tuple Packing and Unpacking: How to pack and unpack tuples.

    Example 8:
    - Tuple Comprehension (Generator Expressions): How to create tuples using generator expressions.
    """
    pass

# End of examples
