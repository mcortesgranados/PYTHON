# FileName: 11_Advanced_Data_Structures.py
# Code Written by Manuel Cortés Granados
# Date: 2024-03-04
# Location: Bogota DC, Colombia
# LinkedIn Profile: https://www.linkedin.com/in/mcortesgranados/

# This Python code demonstrates advanced data structures such as sets, defaultdict, and OrderedDict. 
# It showcases the functionality of each data structure with examples. 
# Additionally, it includes metadata such as the proposed filename, authorship information, date and time, location, 
# and a link to the author's LinkedIn profile for context.

# Advanced Data Structures in Python

# Python provides several advanced data structures that offer specialized functionality for different use cases.

from collections import defaultdict, OrderedDict

# Example of sets
print("Sets:")
set1 = {1, 2, 3, 4, 5}
set2 = {4, 5, 6, 7, 8}

# Union of sets
print("Union:", set1.union(set2))

# Intersection of sets
print("Intersection:", set1.intersection(set2))

# Example of defaultdict
print("\nDefaultDict:")
# Initialize a defaultdict with int as default factory
d = defaultdict(int)
d['a'] = 1
d['b'] = 2
print("DefaultDict:", d)
print("Value of 'c' (default value):", d['c'])

# Example of OrderedDict
print("\nOrderedDict:")
# Initialize an OrderedDict
od = OrderedDict()
od['banana'] = 3
od['apple'] = 2
od['orange'] = 1

# Print OrderedDict items in the order they were added
print("OrderedDict items in insertion order:")
for key, value in od.items():
    print(key, value)
