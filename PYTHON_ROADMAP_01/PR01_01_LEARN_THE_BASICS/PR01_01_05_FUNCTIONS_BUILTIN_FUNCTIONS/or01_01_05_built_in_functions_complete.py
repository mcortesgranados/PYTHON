# Python Built-in Functions Example

# Example 1: print()
# The print() function is used to display text or variables to the console.
# It can take multiple arguments and automatically adds a newline character by default.
print("Hello, world!")

# Example 2: input()
# The input() function is used to get input from the user through the console.
# It displays a prompt to the user and waits for them to enter a value.
name = input("Enter your name: ")
print("Hello,", name)

# Example 3: len()
# The len() function returns the length (number of items) of an object.
# It can be used with strings, lists, tuples, dictionaries, and other iterable objects.
my_list = [1, 2, 3, 4, 5]
length = len(my_list)
print("Length of list:", length)

# Example 4: range()
# The range() function generates a sequence of numbers.
# It can take one, two, or three arguments: start, stop, and step.
# It is commonly used with loops to iterate over a sequence of numbers.
for i in range(5):
    print(i)  # Output: 0 1 2 3 4

# Example 5: sum()
# The sum() function returns the sum of all elements in an iterable.
# It can be used with lists, tuples, and other iterable objects.
numbers = [1, 2, 3, 4, 5]
total = sum(numbers)
print("Sum of numbers:", total)

# Example 6: max() and min()
# The max() function returns the largest element in an iterable.
# The min() function returns the smallest element in an iterable.
# They can be used with lists, tuples, and other iterable objects.
max_value = max(numbers)
min_value = min(numbers)
print("Max value:", max_value)
print("Min value:", min_value)

# Example 7: sorted()
# The sorted() function returns a new sorted list from the elements of any iterable.
# It can take a reverse parameter to sort in descending order.
# It does not modify the original list.
sorted_numbers = sorted(numbers)
print("Sorted numbers:", sorted_numbers)

# Example 8: type()
# The type() function returns the type of an object.
# It is commonly used to check the type of variables.
x = 5
y = "Hello"
print("Type of x:", type(x))  # Output: <class 'int'>
print("Type of y:", type(y))  # Output: <class 'str'>

# Example 9: isinstance()
# The isinstance() function checks if an object is an instance of a specified class.
# It can also accept a tuple of classes to check against.
# It is commonly used for type checking and validation.
result = isinstance(x, int)
print("Is x an integer?", result)  # Output: True

# Example 10: dir()
# The dir() function returns a list of valid attributes and methods of an object.
# It can be used to explore the properties and methods of built-in objects.
dir_result = dir(numbers)
print("Attributes and methods of 'numbers':", dir_result)

# Documenting the Built-in Functions:
def builtin_functions_documentation():
    """
    This function demonstrates various built-in functions in Python.

    Example 1:
    - print(): Used to display text or variables to the console.

    Example 2:
    - input(): Used to get input from the user through the console.

    Example 3:
    - len(): Returns the length of an object (number of items).

    Example 4:
    - range(): Generates a sequence of numbers.

    Example 5:
    - sum(): Returns the sum of all elements in an iterable.

    Example 6:
    - max() and min(): Returns the largest and smallest elements in an iterable.

    Example 7:
    - sorted(): Returns a new sorted list from the elements of any iterable.

    Example 8:
    - type(): Returns the type of an object.

    Example 9:
    - isinstance(): Checks if an object is an instance of a specified class.

    Example 10:
    - dir(): Returns a list of valid attributes and methods of an object.
    """
    pass

# End of examples
