# Python Sets Example

# Example 1: Creating a Set
# Sets in Python are unordered collections of unique elements enclosed within curly braces {}.
# They can contain any immutable data type (integers, strings, floats, tuples, etc.).
# Here's an example of creating a set:
my_set = {1, 2, 3, 4, 5}

# Example 2: Adding Elements to a Set
# You can add elements to a set using the add() method.
# Sets do not allow duplicate elements, so adding an existing element has no effect.
# Here's how you can add elements to the set:
my_set.add(6)   # Add a single element to the set
my_set.update({7, 8})  # Update the set by adding elements from another iterable
print("Updated Set:", my_set)

# Example 3: Removing Elements from a Set
# You can remove elements from a set using methods like remove() and discard().
# Both methods remove the specified element, but remove() raises an error if the element is not present, while discard() does not.
# Here's how you can remove elements from the set:
my_set.remove(5)  # Remove element 5 from the set
my_set.discard(10)  # Trying to remove non-existent element 10 using discard (no error raised)
print("Set after Removal:", my_set)

# Example 4: Set Operations
# Python sets support various mathematical operations such as union, intersection, difference, and symmetric difference.
# Here's how you can perform set operations:
set1 = {1, 2, 3, 4, 5}
set2 = {4, 5, 6, 7, 8}

union_set = set1.union(set2)               # Union of set1 and set2
intersection_set = set1.intersection(set2) # Intersection of set1 and set2
difference_set = set1.difference(set2)     # Difference of set1 - set2
symmetric_difference_set = set1.symmetric_difference(set2)  # Symmetric difference of set1 and set2

print("Union Set:", union_set)
print("Intersection Set:", intersection_set)
print("Difference Set:", difference_set)
print("Symmetric Difference Set:", symmetric_difference_set)

# Example 5: Set Methods
# Python provides several built-in methods to manipulate sets.
# Here are some commonly used set methods:
# - len(): Returns the number of elements in the set.
# - clear(): Removes all elements from the set.
# - copy(): Returns a shallow copy of the set.
# Here's how you can use these methods:
print("Number of Elements in set1:", len(set1))
set1.clear()   # Clear all elements from set1
set2_copy = set2.copy()  # Create a shallow copy of set2
print("Cleared set1:", set1)
print("Copied set2:", set2_copy)

# Example 6: Set Comprehension
# Set comprehension is a concise way to create sets in Python.
# It allows you to create a new set by applying an expression to each item in an existing iterable.
# Here's how you can use set comprehension to create a set of squares:
squares_set = {x**2 for x in range(1, 6)}
print("Squares Set:", squares_set)

# Documenting the Sets:
def sets_documentation():
    """
    This function demonstrates various aspects of sets in Python.

    Example 1:
    - Creating a Set: How to define a set in Python.

    Example 2:
    - Adding Elements to a Set: How to add elements to a set.

    Example 3:
    - Removing Elements from a Set: How to remove elements from a set.

    Example 4:
    - Set Operations: Common mathematical operations on sets.

    Example 5:
    - Set Methods: Commonly used methods to manipulate sets.

    Example 6:
    - Set Comprehension: How to create sets using set comprehension.
    """
    pass

# End of examples
