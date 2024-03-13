# String variables
string1 = "Hello"
string2 = "world"

# Concatenation
concatenated_string = string1 + " " + string2
print("Concatenation:", concatenated_string)

# String repetition
repeated_string = string1 * 3
print("String repetition:", repeated_string)

# String length
length = len(concatenated_string)
print("String length:", length)

# String slicing
substring = concatenated_string[3:8]
print("Substring:", substring)

# String lowercase
lowercased_string = concatenated_string.lower()
print("Lowercase:", lowercased_string)

# String uppercase
uppercased_string = concatenated_string.upper()
print("Uppercase:", uppercased_string)

# String formatting
formatted_string = "My name is {} and I am {} years old".format("Alice", 30)
print("Formatted string:", formatted_string)

# String interpolation (f-strings, available in Python 3.6+)
name = "Bob"
age = 25
interpolated_string = f"My name is {name} and I am {age} years old"
print("Interpolated string:", interpolated_string)

# String splitting
split_string = concatenated_string.split(" ")
print("String splitting:", split_string)

# String stripping
whitespace_string = "   This is a string with whitespace   "
stripped_string = whitespace_string.strip()
print("Stripped string:", stripped_string)

# String replacement
replaced_string = concatenated_string.replace("world", "Python")
print("String replacement:", replaced_string)

# String checking
check_start = concatenated_string.startswith("Hello")
check_end = concatenated_string.endswith("world")
print("Starts with 'Hello'?", check_start)
print("Ends with 'world'?", check_end)
