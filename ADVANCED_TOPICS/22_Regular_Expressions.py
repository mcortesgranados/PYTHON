# FileName: 22_Regular_Expressions.py
# Code Written by Manuel Cortés Granados
# Date: 2024-03-04
# Location: Bogota DC, Colombia
# LinkedIn Profile: https://www.linkedin.com/in/mcortesgranados/

# Regular Expressions in Python

# Regular expressions (regex) provide a powerful way to search, manipulate, and validate strings.
# Python's re module provides support for regular expressions.

import re

# Example: Search for a pattern in a string
text = "Hello, this is a sample text with some numbers like 12345 and special characters !@#$%"
pattern = r'\d+'  # Match one or more digits
matches = re.findall(pattern, text)
print("Matches:", matches)

# Example: Replace pattern in a string
replaced_text = re.sub(pattern, "NUMBER", text)
print("Replaced Text:", replaced_text)
