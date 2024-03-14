# In this code:

# The union() method returns a set containing all the distinct elements from both sets.
# The intersection() method returns a set containing elements that are common to both sets.
# The difference() method returns a set containing elements that are present in the first set but not in the second set.
# The symmetric_difference() method returns a set containing elements that are present in either set but not in both.
# The issubset() method checks if one set is a subset of another.
# The issuperset() method checks if one set is a superset of another.
# These set operations are useful for comparing and manipulating sets in Python.

# Creating sets
set1 = {1, 2, 3, 4, 5}
set2 = {4, 5, 6, 7, 8}

# Union: Elements that are in either set1 or set2 (or both)
union_set = set1.union(set2)
print("Union set:", union_set)

# Intersection: Elements that are common to both set1 and set2
intersection_set = set1.intersection(set2)
print("Intersection set:", intersection_set)

# Difference: Elements that are in set1 but not in set2
difference_set1 = set1.difference(set2)
print("Difference set (set1 - set2):", difference_set1)

# Difference: Elements that are in set2 but not in set1
difference_set2 = set2.difference(set1)
print("Difference set (set2 - set1):", difference_set2)

# Symmetric Difference: Elements that are in either set1 or set2 but not in both
symmetric_difference_set = set1.symmetric_difference(set2)
print("Symmetric difference set:", symmetric_difference_set)

# Subset check: Check if set1 is a subset of set2
is_subset = set1.issubset(set2)
print("Is set1 a subset of set2?", is_subset)

# Superset check: Check if set2 is a superset of set1
is_superset = set2.issuperset(set1)
print("Is set2 a superset of set1?", is_superset)
