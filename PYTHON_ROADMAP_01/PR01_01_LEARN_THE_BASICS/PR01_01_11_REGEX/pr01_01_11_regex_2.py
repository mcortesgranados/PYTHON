import re

# Python Regular Expressions Example

# Example 1: Matching Patterns with re.match()
# The re.match() function is used to match patterns at the beginning of a string.
# It returns a match object if the pattern is found, or None otherwise.
# Here's an example of using re.match() to match a pattern at the beginning of a string:
pattern = r'hello'
text = 'hello world'
match = re.match(pattern, text)
if match:
    print("Match found:", match.group())
else:
    print("No match")

# Example 2: Searching for Patterns with re.search()
# The re.search() function searches for a pattern in the entire string.
# It returns a match object if the pattern is found, or None otherwise.
# Here's an example of using re.search() to search for a pattern in a string:
pattern = r'world'
text = 'hello world'
search = re.search(pattern, text)
if search:
    print("Pattern found at index:", search.start())
else:
    print("Pattern not found")

# Example 3: Finding All Matches with re.findall()
# The re.findall() function finds all occurrences of a pattern in a string.
# It returns a list of all matches found.
# Here's an example of using re.findall() to find all occurrences of a pattern in a string:
pattern = r'\d+'  # Matches one or more digits
text = 'I have 10 apples and 20 oranges'
matches = re.findall(pattern, text)
print("All matches:", matches)

# Example 4: Splitting Strings with re.split()
# The re.split() function splits a string based on a specified pattern.
# It returns a list of substrings.
# Here's an example of using re.split() to split a string based on whitespace:
pattern = r'\s+'  # Matches one or more whitespace characters
text = 'hello world python'
split_result = re.split(pattern, text)
print("Split result:", split_result)

# Example 5: Replacing Patterns with re.sub()
# The re.sub() function replaces occurrences of a pattern in a string with a specified replacement string.
# It returns a new string with replacements made.
# Here's an example of using re.sub() to replace digits with 'X' in a string:
pattern = r'\d'  # Matches any digit
text = 'I have 10 apples and 20 oranges'
replacement = 'X'
new_text = re.sub(pattern, replacement, text)
print("Replaced text:", new_text)

# Example 6: Using Capture Groups
# Capture groups allow you to extract parts of a matched pattern.
# They are defined using parentheses () in the pattern.
# Here's an example of using capture groups to extract username and domain from an email address:
pattern = r'(\w+)@(\w+\.\w+)'  # Matches email addresses
text = 'Email: user@example.com'
match = re.search(pattern, text)
if match:
    username = match.group(1)
    domain = match.group(2)
    print("Username:", username)
    print("Domain:", domain)

# Documenting the Regular Expressions:
def regex_documentation():
    """
    This function demonstrates various aspects of regular expressions in Python.

    Example 1:
    - Matching Patterns with re.match(): How to match patterns at the beginning of a string.

    Example 2:
    - Searching for Patterns with re.search(): How to search for patterns in a string.

    Example 3:
    - Finding All Matches with re.findall(): How to find all occurrences of a pattern in a string.

    Example 4:
    - Splitting Strings with re.split(): How to split a string based on a pattern.

    Example 5:
    - Replacing Patterns with re.sub(): How to replace occurrences of a pattern in a string.

    Example 6:
    - Using Capture Groups: How to use capture groups to extract parts of a matched pattern.
    """
    pass

# End of examples
