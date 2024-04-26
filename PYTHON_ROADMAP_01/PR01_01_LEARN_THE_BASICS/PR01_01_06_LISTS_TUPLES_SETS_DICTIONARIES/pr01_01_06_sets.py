# This program demonstrates working with sets in Python

# Sets (unordered collections of unique elements)

# Sets are used to store collections of elements where order doesn't matter 
# and elements must be unique (no duplicates allowed). Sets can contain 
# various data types like strings, integers, or even custom objects.

# Defining a set
unique_fruits = {"apple", "banana", "orange", "banana"}  # Duplicate "banana" will be ignored

# Accessing elements in sets is not done by index (elements are unordered)
# for item in unique_fruits:  # Looping through elements is a common approach
#   print(item)

# Checking membership
if "apple" in unique_fruits:
  print("Apple is in the set.")

# Adding elements to sets
unique_fruits.add("mango")  # Add a new element

print("Updated set with mango:", unique_fruits)

# Removing elements
unique_fruits.remove("orange")  # Remove "orange" (careful, raises error if not present)

print("Set without orange:", unique_fruits)

# Removing elements safely (avoiding errors)
if "grape" in unique_fruits:
  unique_fruits.remove("grape")  # Only remove if it exists
else:
  print("Grape is not in the set, so nothing is removed.")

print("Set (unchanged, grape not present):", unique_fruits)

# Set operations (demonstrated with another set)
colors = {"red", "green", "blue"}

# Union - combines elements from both sets (keeps unique elements)
all_items = unique_fruits.union(colors)
print("Union of fruits and colors:", all_items)

# Intersection - elements present in both sets
common_elements = unique_fruits.intersection(colors)
print("Intersection (common elements):", common_elements)

# Difference - elements in first set but not in the second
fruits_not_colors = unique_fruits.difference(colors)
print("Difference (fruits not in colors):", fruits_not_colors)
