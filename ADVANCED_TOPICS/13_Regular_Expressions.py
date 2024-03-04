# FileName: 13_Regular_Expressions.py
# Code Written by Manuel Cortés Granados
# Date: 2024-03-04
# Location: Bogota DC, Colombia
# LinkedIn Profile: https://www.linkedin.com/in/mcortesgranados/

# Regular Expressions in Python

# Regular expressions (regex) are a powerful tool for matching patterns in text data. 
# Python's re module provides support for regular expressions.

import re

# Example of using regular expressions to search for patterns in text
text = "Hello, my email address is example@email.com"
pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
emails = re.findall(pattern, text)
print("Email addresses found:", emails)
