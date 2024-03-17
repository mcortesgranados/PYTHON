
# Regular expressions (regex) are a powerful tool for pattern matching and string manipulation. In Python, the re module provides support for
#  regular expressions. Here's an introduction to using regular expressions in Python:

# Matching Patterns:

# re.match(pattern, string): Attempts to match the pattern at the beginning of the string.

import re

pattern = r'hello'
string = 'hello world'
match = re.match(pattern, string)
if match:
    print('Pattern matched')


# Searching Patterns:

# re.search(pattern, string): Searches for the first occurrence of the pattern in the string.

import re

pattern = r'world'
string = 'hello world'
match = re.search(pattern, string)
if match:
    print('Pattern found')


# Finding All Matches:

# re.findall(pattern, string): Finds all occurrences of the pattern in the string and returns them as a list.
    
import re

pattern = r'o'
string = 'hello world'
matches = re.findall(pattern, string)
print(matches)


# Splitting Strings:

# re.split(pattern, string): Splits the string based on the occurrences of the pattern

import re

pattern = r'\s+'  # Split on whitespace
string = 'hello   world'
parts = re.split(pattern, string)
print(parts)


# Replacing Patterns:

# re.sub(pattern, replacement, string): Replaces occurrences of the pattern in the string with the replacement string.

import re

pattern = r'\s+'  # Replace whitespace with a single space
string = 'hello   world'
new_string = re.sub(pattern, ' ', string)
print(new_string)


