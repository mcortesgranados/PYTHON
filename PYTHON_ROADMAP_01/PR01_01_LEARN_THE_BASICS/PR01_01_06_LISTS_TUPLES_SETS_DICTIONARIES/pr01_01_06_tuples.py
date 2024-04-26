# This program demonstrates working with tuples in Python

# Tuples (immutable ordered collections)

# Tuples are ordered collections of elements similar to lists. 
# However, unlike lists, tuples are immutable, meaning their elements 
# cannot be changed after creation.

# Defining a tuple
fruits = ("apple", "banana", "orange")

# Accessing elements by index
first_fruit = fruits[0]  # Accessing by index (starts from 0)

print("The first fruit is:", first_fruit)

# Tuples can hold elements of different data types
person = ("Alice", 30, "New York")  # Name, age, city

# Immutability - attempting to modify will cause an error
# try:
#   fruits[1] = "mango"  # This will raise a TypeError
# except TypeError:
#   print("Tuples are immutable!")

# You can create a new tuple with modifications
updated_fruits = fruits + ("mango",)  # Add "mango" as a new element

print("Updated fruits (new tuple):", updated_fruits)
