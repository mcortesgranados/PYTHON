"""
Regular Expressions in Python (Comprehensive Demonstration)

This script comprehensively demonstrates regular expressions (regex) 
for pattern matching and text manipulation in Python.
"""

import re

# Matching simple patterns

text = "This is a sample text string."

# Match the word "text"
match = re.search("text", text)

if match:
  print("Matched word:", match.group())  # Access the matched text
else:
  print("Word 'text' not found.")

# Match any digit (0-9)
match = re.search("\d", text)

if match:
  print("Matched a digit:", match.group())
else:
  print("No digits found.")

# Matching with character classes

# Match any lowercase letter (a-z)
match = re.search("[a-z]", text)

if match:
  print("Matched a lowercase letter:", match.group())

# Match any whitespace character (\s)
match = re.search("\s", text)

if match:
  print("Matched whitespace:", match.group())

# Matching repetitions

# Match "is" repeated exactly twice
match = re.search("is{2}", text)

if match:
  print("Matched 'is' repeated twice:", match.group())

# Match one or more occurrences of "am"
match = re.search("am+", text)

if match:
  print("Matched 'am' one or more times:", match.group())

# Matching with groups

# Extract the first word
match = re.search("^(\w+)", text)  # ^ for beginning, \w+ for word characters

if match:
  print("First word:", match.group(1))  # Group 1 captures the matched word

# Find all email addresses (simplified example)
emails = re.findall(r"[\w\.]+@[\w\.]+\.\w{2,}", text)

if emails:
  print("Found email addresses:", emails)
else:
  print("No email addresses found.")

# Replacing patterns

new_text = re.sub(r"\sis\s", " was ", text)  # Replace "is" with "was"
print("Text with replaced word:", new_text)

# Advanced techniques

# Matching with flags (caseless)
match = re.search("SAMPLE", text, flags=re.IGNORECASE)

if match:
  print("Matched 'SAMPLE' (case-insensitive):", match.group())

# Finding all non-whitespace characters
non_whitespace = re.findall(r"\S+", text)

if non_whitespace:
  print("All non-whitespace words:", non_whitespace)

# Key Points:

* Regular expressions provide a powerful tool for pattern matching and text manipulation.
* Metacharacters and character classes offer flexibility for defining patterns.
* Repetitions allow matching patterns that occur a specific number of times.
* Grouping captures parts of the matched pattern for later use.
* Substitutions enable replacing matched patterns with new text.
* Flags can modify regex behavior (e.g., case-insensitive matching).

"""This script concludes the demonstration of regular expressions."""
