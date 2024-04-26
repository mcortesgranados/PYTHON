# This program demonstrates working with sets in Python

# Sets (unordered collections of unique elements)

# Sets are collections of unique elements, meaning they cannot contain 
# duplicate values. Elements must be hashable (like strings, numbers, 
# or tuples), meaning they can be used as keys in dictionaries. 
# Sets are unordered, so the order of elements in a set does not matter.

# Defining a set
fruits = {"apple", "banana", "orange", "banana"}  # Duplicate "banana" will be ignored

print("Set of fruits:", fruits)

# Checking for membership
if "apple" in fruits:
  print("Apple is in the set.")
else:
  print("Apple is not in the set.")

# Adding elements
fruits.add("grape")  # Add a new element

print("Updated set with grape:", fruits)

# Removing elements
try:
  fruits.remove("orange")  # Remove an element (raises KeyError if not found)
  print("Orange removed successfully.")
except KeyError:
  print("Orange not found in the set.")

fruits.discard("mango")  # Attempts to remove "mango" (doesn't raise error if not found)

print("Set with orange removed and mango discard attempted:", fruits)

# Set operations
# - Union (combining elements)
other_fruits = {"mango", "kiwi", "apple"}
combined_fruits = fruits.union(other_fruits)
print("Combined fruits (fruits union other_fruits):", combined_fruits)

# - Intersection (common elements)
common_fruits = fruits.intersection(other_fruits)
print("Common fruits (fruits intersection other_fruits):", common_fruits)

# - Difference (elements in first set but not second)
unique_in_fruits = fruits.difference(other_fruits)
print("Fruits unique to the first set (fruits difference other_fruits):", unique_in_fruits)

# - Symmetric difference (elements in either set but not both)
all_unique = fruits.symmetric_difference(other_fruits)
print("Elements in one set but not the other (fruits symmetric_difference other_fruits):", all_unique)

# Modifying a set (careful, may not preserve order)
fruits.update(other_fruits)  # Update the set with elements from other_fruits (may cause duplicates)
print("Fruits updated with other_fruits (careful, order may not be preserved):", fruits)

# Clearing the set
fruits.clear()
print("Set cleared:", fruits)

# Looping through elements
for fruit in fruits:  # After clearing, the loop won't iterate since the set is empty
  print(fruit)

# Converting between sets and lists
fruits_list = list(combined_fruits)  # Convert set to list (order may not be preserved)

print("List converted from the combined set:", fruits_list)
