# 
# Advanced string formatting techniques in Python allow for more complex and flexible string manipulation. Here are some techniques:

# Format Method:

# Use the format() method to insert values into a string.

name = "Alice"
age = 30
message = "My name is {} and I am {} years old.".format(name, age)
print(message)


# f-strings (Python 3.6+):

# Use f-strings to embed expressions inside strings.

name = "Alice"
age = 30
message = f"My name is {name} and I am {age} years old."
print(message)


# Alignment and Padding:

# Use alignment and padding specifiers within curly braces {}.

value = 42
print(f"Value: {value:10}")  # Right-aligned, padded with spaces
print(f"Value: {value:<10}")  # Left-aligned, padded with spaces
print(f"Value: {value:^10}")  # Center-aligned, padded with spaces


# Precision and Number Formatting:

# Use precision and number formatting specifiers.

pi = 3.14159
print(f"Pi: {pi:.2f}")    # Two decimal places
print(f"Pi: {pi:.4f}")    # Four decimal places
print(f"Pi: {pi:.2e}")    # Scientific notation with two decimal places


# Formatted String Literals (f-strings) with Expressions:

# Perform expressions within f-strings.

a = 10
b = 20
print(f"The sum of {a} and {b} is {a + b}")


# Formatted String Literals with Dictionaries:

# Use dictionaries with f-strings for more complex formatting.

person = {'name': 'Alice', 'age': 30}
print(f"Name: {person['name']}, Age: {person['age']}")





