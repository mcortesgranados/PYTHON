# Python Lists Example

# Example 1: Creating a List
# Lists in Python are ordered collections of items enclosed within square brackets [].
# They can contain any data type (integers, strings, floats, other lists, etc.).
# Here's an example of creating a list:
my_list = [1, 2, 3, 4, 5]

# Example 2: Accessing Elements in a List
# You can access individual elements in a list using indexing.
# Indexing starts from 0 for the first element, 1 for the second, and so on.
# Here's how you can access elements in the list:
first_element = my_list[0]
second_element = my_list[1]
print("First Element:", first_element)
print("Second Element:", second_element)

# Example 3: Slicing Lists
# You can slice lists to extract a subset of elements using a start and end index.
# Slicing returns a new list containing the specified range of elements.
# Here's how you can slice a list:
subset = my_list[1:4]  # Extract elements from index 1 to index 3 (end index is exclusive)
print("Subset:", subset)

# Example 4: Modifying Elements in a List
# You can modify individual elements or slices of a list.
# Lists are mutable, meaning their elements can be changed after they are created.
# Here's how you can modify elements in the list:
my_list[2] = 10  # Update the value at index 2
print("Modified List:", my_list)

# Example 5: Adding Elements to a List
# You can add elements to a list using various methods such as append(), insert(), and extend().
# Here's how you can add elements to the list:
my_list.append(6)       # Append a single element to the end of the list
my_list.insert(0, 0)    # Insert an element at a specific position (index 0 in this case)
my_list.extend([7, 8])  # Extend the list by appending elements from another iterable
print("Updated List:", my_list)

# Example 6: Removing Elements from a List
# You can remove elements from a list using methods like remove(), pop(), and clear().
# Here's how you can remove elements from the list:
my_list.remove(5)  # Remove the first occurrence of value 5
popped_element = my_list.pop(1)  # Remove and return the element at index 1
my_list.clear()   # Remove all elements from the list
print("Empty List:", my_list)
print("Popped Element:", popped_element)

# Example 7: List Methods
# Python provides several built-in methods to manipulate lists.
# Here are some commonly used list methods:
# - index(): Returns the index of the first occurrence of a value in the list.
# - count(): Returns the number of occurrences of a value in the list.
# - sort(): Sorts the elements of the list in ascending order.
# - reverse(): Reverses the order of elements in the list.
# - copy(): Returns a shallow copy of the list.
# Here's how you can use these methods:
numbers = [3, 1, 4, 1, 5, 9, 2, 6, 5]
index = numbers.index(9)  # Find the index of value 9
count = numbers.count(5)  # Count the number of occurrences of value 5
numbers.sort()            # Sort the list in ascending order
numbers.reverse()         # Reverse the order of elements
print("Index of 9:", index)
print("Count of 5:", count)
print("Sorted List:", numbers)

# Example 8: List Comprehension
# List comprehension is a concise way to create lists in Python.
# It allows you to create a new list by applying an expression to each item in an existing iterable.
# Here's how you can use list comprehension to create a list of squares:
squares = [x**2 for x in range(1, 6)]
print("Squares:", squares)

# Documenting the Lists:
def lists_documentation():
    """
    This function demonstrates various aspects of lists in Python.

    Example 1:
    - Creating a List: How to define a list in Python.

    Example 2:
    - Accessing Elements in a List: How to access elements using indexing.

    Example 3:
    - Slicing Lists: How to extract a subset of elements using slicing.

    Example 4:
    - Modifying Elements in a List: How to modify elements in a list.

    Example 5:
    - Adding Elements to a List: How to add elements to a list.

    Example 6:
    - Removing Elements from a List: How to remove elements from a list.

    Example 7:
    - List Methods: Commonly used methods to manipulate lists.

    Example 8:
    - List Comprehension: How to create lists using list comprehension.
    """
    pass

# End of examples
