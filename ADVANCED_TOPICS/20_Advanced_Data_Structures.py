# FileName: 20_Advanced_Data_Structures.py
# Code Written by Manuel Cortés Granados
# Date: 2024-03-04
# Location: Bogota DC, Colombia
# LinkedIn Profile: https://www.linkedin.com/in/mcortesgranados/

# Advanced Data Structures in Python

# Python provides several advanced data structures that offer additional functionality and flexibility compared to basic data structures.

from collections import defaultdict, OrderedDict

# Set: Unordered collection of unique elements
num_set = {1, 2, 3, 4, 5}
print("Set:", num_set)

# defaultdict: Dictionary subclass that calls a factory function to supply missing values
# It is useful for handling missing keys gracefully
num_dict = defaultdict(int)
num_dict['one'] = 1
num_dict['two'] = 2
print("DefaultDict:", num_dict)
print("Accessing missing key 'three':", num_dict['three'])  # Returns default value for missing key (0 for int)

# OrderedDict: Dictionary subclass that remembers the order in which its contents are added
# Useful for creating dictionaries where order matters
ordered_dict = OrderedDict()
ordered_dict['a'] = 1
ordered_dict['b'] = 2
ordered_dict['c'] = 3
print("OrderedDict:", ordered_dict)
print("Keys in OrderedDict:", list(ordered_dict.keys()))
