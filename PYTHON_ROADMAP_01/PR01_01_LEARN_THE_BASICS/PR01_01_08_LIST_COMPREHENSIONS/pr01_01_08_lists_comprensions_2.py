# Python List Comprehension Example

# Example 1: Creating a List using List Comprehension
# List comprehension provides a concise way to create lists in Python.
# It allows you to apply an expression to each item in an existing iterable.
# Here's an example of creating a list of squares using list comprehension:
squares = [x**2 for x in range(1, 6)]
# Explanation: This expression generates a list containing the square of each number from 1 to 5.
# The syntax is [expression for item in iterable].
# In this case, the expression is x**2, the item is x, and the iterable is range(1, 6).
print("Squares:", squares)

# Example 2: Filtering Elements with List Comprehension
# List comprehension can also be used to filter elements from an existing iterable.
# You can include a conditional expression to filter items based on a condition.
# Here's an example of filtering even numbers from a list:
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
even_numbers = [x for x in numbers if x % 2 == 0]
# Explanation: This expression generates a list containing only the even numbers from the original list.
# The conditional expression 'if x % 2 == 0' filters out odd numbers.
print("Even Numbers:", even_numbers)

# Example 3: Nested List Comprehension
# You can use nested list comprehension to create lists of lists.
# This allows you to generate a 2D list or apply multiple levels of transformation.
# Here's an example of creating a 3x3 identity matrix using nested list comprehension:
identity_matrix = [[1 if i == j else 0 for j in range(3)] for i in range(3)]
# Explanation: This expression generates a 2D list where each inner list represents a row of the identity matrix.
# The conditional expression '1 if i == j else 0' sets the diagonal elements to 1 and others to 0.
print("Identity Matrix:")
for row in identity_matrix:
    print(row)

# Example 4: Using List Comprehension with String Manipulation
# List comprehension is not limited to numerical operations; you can also perform string manipulation.
# Here's an example of converting a list of strings to uppercase using list comprehension:
words = ["hello", "world", "python", "list", "comprehension"]
uppercase_words = [word.upper() for word in words]
# Explanation: This expression generates a list where each string is converted to uppercase.
print("Uppercase Words:", uppercase_words)

# Example 5: List Comprehension with Dictionary
# You can use list comprehension to create lists from dictionary elements.
# Here's an example of creating a list of keys from a dictionary:
my_dict = {"a": 1, "b": 2, "c": 3}
keys_list = [key for key in my_dict]
# Explanation: This expression generates a list containing all keys from the dictionary.
print("Keys List:", keys_list)

# Example 6: List Comprehension with Conditionals and Nested Iterables
# List comprehension supports complex expressions with conditionals and nested iterables.
# Here's an example of creating a list of tuples using conditional expressions and nested iteration:
pairs = [(x, y) for x in range(3) for y in range(3) if x != y]
# Explanation: This expression generates a list of tuples where each tuple contains two different numbers from 0 to 2.
# The conditional expression 'if x != y' ensures that the numbers in each tuple are different.
print("Pairs:", pairs)

# Documenting the List Comprehension:
def list_comprehension_documentation():
    """
    This function demonstrates various aspects of list comprehension in Python.

    Example 1:
    - Creating a List using List Comprehension: How to create lists using list comprehension.

    Example 2:
    - Filtering Elements with List Comprehension: How to filter elements from an iterable using list comprehension.

    Example 3:
    - Nested List Comprehension: How to use nested list comprehension to create 2D lists.

    Example 4:
    - Using List Comprehension with String Manipulation: How to perform string manipulation using list comprehension.

    Example 5:
    - List Comprehension with Dictionary: How to create lists from dictionary elements using list comprehension.

    Example 6:
    - List Comprehension with Conditionals and Nested Iterables: How to use complex expressions with conditionals and nested iterables.
    """
    pass

# End of examples
