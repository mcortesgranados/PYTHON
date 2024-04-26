# This program demonstrates working with sets in Python

# Sets (unordered collections of unique elements)

# Sets are collections of unique elements, meaning they cannot contain 
# duplicate values. Elements must be hashable (like strings, numbers, 
# or tuples), meaning they can be used as keys in dictionaries. 
# Sets are unordered, so the order of elements in a set does not matter.

# Defining a set
fruits = {"apple", "banana", "orange", "banana"}  # Duplicate "banana" will be ignored

# Accessing elements (not by index)
# Sets don't support direct element access by index since they are unordered.

# Checking for membership
if "apple" in fruits:
  print("Apple is in the set.")
else:
  print("Apple is not in the set.")

# Adding elements
fruits.add("grape")  # Add a new element

print("Updated set with grape:", fruits)

# Removing elements
fruits.remove("orange")  # Remove an element (raises KeyError if not found)
fruits.discard("mango")  # Attempts to remove "mango" (doesn't raise error if not found)

print("Set with orange removed and mango discard attempted:", fruits)

# Set operations (demonstrated without modification)
# fruits.union(other_set) - Returns a new set with elements from both sets (combining)
# fruits.intersection(other_set) - Returns a new set with elements common to both sets
# fruits.difference(other_set) - Returns a new set with elements in the first set but not in the second
# fruits.clear() - Removes all elements from the set

# Looping through elements
for fruit in fruits:
  print(fruit)

# Converting between sets and lists
fruits_list = list(fruits)  # Convert set to list (order may not be preserved)

print("List converted from the set:", fruits_list)
