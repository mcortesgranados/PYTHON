# Python Type Casting Example

# Example 1: Implicit Type Casting (Automatic Conversion)
# Python automatically converts data from one type to another when needed, known as implicit type casting.
# For example, when adding an integer and a float, Python converts the integer to a float before performing the addition.
x = 10      # integer
y = 3.14    # float
result = x + y
print("Result:", result)  # Output: 13.14 (integer 10 is implicitly converted to float before addition)

# Example 2: Explicit Type Casting (Manual Conversion)
# Explicit type casting is the process of converting data from one type to another using built-in functions.
# Python provides built-in functions for explicit type casting: int(), float(), str(), etc.
# Here's an example of explicit type casting:
num_str = "123"     # string containing digits
num_int = int(num_str)   # convert string to integer
print("Integer:", num_int)  # Output: 123

# Example 3: Type Casting Between Numerical Types
# You can convert between numerical types like int, float, and complex using explicit type casting.
# Here's an example:
float_num = 3.14
int_num = int(float_num)   # convert float to integer (truncates decimal part)
print("Integer:", int_num)  # Output: 3

# Example 4: Type Casting Between Strings and Numerical Types
# You can convert between strings and numerical types using explicit type casting functions like int(), float(), and str().
# Here's an example:
num_str = "42"
num_int = int(num_str)     # convert string to integer
num_float = float(num_str) # convert string to float
print("Integer:", num_int)   # Output: 42
print("Float:", num_float)   # Output: 42.0

# Example 5: Type Casting with Boolean Values
# Boolean values (True and False) can be cast to integers (1 and 0) and vice versa.
# Here's an example:
bool_value = True
int_value = int(bool_value)     # convert boolean to integer (True becomes 1)
print("Integer:", int_value)    # Output: 1

# Example 6: Handling Type Casting Errors
# Type casting errors can occur if the data cannot be converted to the desired type.
# For example, trying to convert a string containing letters to an integer will raise a ValueError.
# Here's an example:
invalid_str = "abc"
try:
    invalid_int = int(invalid_str)  # Trying to convert string containing letters to integer
    print("Integer:", invalid_int)
except ValueError as e:
    print("Error:", e)   # Output: Error: invalid literal for int() with base 10: 'abc'

# Documenting the Type Casting:
def type_casting_documentation():
    """
    This function demonstrates various aspects of type casting in Python.

    Example 1:
    - Implicit Type Casting: Automatic conversion of data from one type to another.

    Example 2:
    - Explicit Type Casting: Manual conversion of data using built-in functions.

    Example 3:
    - Type Casting Between Numerical Types: Converting between int, float, and complex types.

    Example 4:
    - Type Casting Between Strings and Numerical Types: Converting between strings and numerical types.

    Example 5:
    - Type Casting with Boolean Values: Converting boolean values to integers and vice versa.

    Example 6:
    - Handling Type Casting Errors: Dealing with errors that occur during type casting.
    """
    pass

# End of examples
