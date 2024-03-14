# Sets in Python are unordered collections of unique elements. Below is a Python code sample demonstrating the creation and usage of sets:

# In this code:

# Sets are created using curly braces {} or the set() constructor.
# Elements are added to a set using the add() method.
# Elements are removed from a set using the remove() method.
# Set operations such as union, intersection, and difference are performed using set methods or operators (|, &, -).
# Sets are commonly used for tasks like removing duplicates from a sequence, checking for membership, and performing set operations like union and intersection.

# Creating a set
my_set = {1, 2, 3, 4, 5}
print("Set:", my_set)

# Adding elements to a set
my_set.add(6)
my_set.add(7)
print("Set after adding elements:", my_set)

# Removing elements from a set
my_set.remove(3)
print("Set after removing element 3:", my_set)

# Set operations: union, intersection, difference
set1 = {1, 2, 3, 4}
set2 = {3, 4, 5, 6}

# Union
union_set = set1.union(set2)
print("Union set:", union_set)

# Intersection
intersection_set = set1.intersection(set2)
print("Intersection set:", intersection_set)

# Difference
difference_set = set1.difference(set2)
print("Difference set (set1 - set2):", difference_set)
