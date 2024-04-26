# Python Arrays Example

# Example 1: Creating an Array (List)
# In Python, arrays are commonly implemented using lists.
# Lists can hold elements of any data type and offer various built-in methods.
# Here's an example of creating a list to represent an array:

numbers = [1, 2, 3, 4, 5]
# Explanation: This creates a list named 'numbers' containing integers.
# Lists are enclosed in square brackets [] and elements are separated by commas.

print("Array (List):", numbers)

# Example 2: Accessing Elements of an Array
# You can access elements of an array using indexing.
# Indexing starts at 0 for the first element.
# Here's an example of accessing elements of the 'numbers' array:

print("Element at index 0:", numbers[0])
print("Element at index 3:", numbers[3])

# Example 3: Slicing Arrays
# Slicing allows you to extract a subset of elements from an array.
# It's done using the syntax array[start:end].
# Here's an example of slicing the 'numbers' array:

subset = numbers[1:4]
# Explanation: This extracts elements at index 1, 2, and 3 (end index is exclusive).
# The subset includes elements 2, 3, and 4.

print("Subset:", subset)

# Example 4: Modifying Elements of an Array
# You can modify elements of an array by assigning new values to specific indices.
# Here's an example of modifying elements of the 'numbers' array:

numbers[2] = 10
# Explanation: This modifies the element at index 2 to the value 10.

print("Modified Array:", numbers)

# Example 5: Adding Elements to an Array
# You can add elements to the end of an array using the append() method.
# Here's an example of adding elements to the 'numbers' array:

numbers.append(6)
# Explanation: This appends the value 6 to the end of the 'numbers' array.

print("Array after appending:", numbers)

# Example 6: Removing Elements from an Array
# You can remove elements from an array using various methods like pop(), remove(), or del.
# Here's an example of removing an element from the 'numbers' array:

removed_element = numbers.pop(1)
# Explanation: This removes the element at index 1 (2nd element) from the 'numbers' array.

print("Removed Element:", removed_element)
print("Array after removing:", numbers)

# Documenting the Arrays:
def arrays_documentation():
    """
    This function demonstrates various aspects of arrays (implemented using lists) in Python.

    Example 1:
    - Creating an Array (List): How to create a list to represent an array.

    Example 2:
    - Accessing Elements of an Array: How to access elements of an array using indexing.

    Example 3:
    - Slicing Arrays: How to extract a subset of elements from an array using slicing.

    Example 4:
    - Modifying Elements of an Array: How to modify elements of an array by assigning new values.

    Example 5:
    - Adding Elements to an Array: How to add elements to the end of an array using the append() method.

    Example 6:
    - Removing Elements from an Array: How to remove elements from an array using methods like pop(), remove(), or del.
    """
    pass

# End of examples
