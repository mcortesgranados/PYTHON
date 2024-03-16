# Sorting lists in Python is a common operation that arranges the elements of a list in a specific order. 
# Python provides built-in functions and methods for sorting lists, giving you flexibility in sorting based on various criteria. 
# Here are the common ways to sort lists in Python:

# Using the sorted() Function:

# The sorted() function returns a new sorted list from the elements of any iterable.
# It does not modify the original list but returns a new sorted list.
# Syntax: sorted(iterable, key=None, reverse=False)
# iterable: The iterable (e.g., list, tuple, set) to be sorted.
# key (optional): A function that specifies a custom sorting order. It is applied to each element before comparison.
# reverse (optional): A boolean value that specifies whether to sort in descending order (True) or ascending order (False, default).
# Example:

numbers = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
sorted_numbers = sorted(numbers)
print(sorted_numbers)  # Output: [1, 1, 2, 3, 3, 4, 5, 5, 6, 9]


# Using the sort() Method:

# The sort() method sorts the elements of a list in place (modifies the original list).
# Syntax: list.sort(key=None, reverse=False)
# key (optional): A function that specifies a custom sorting order. It is applied to each element before comparison.
# reverse (optional): A boolean value that specifies whether to sort in descending order (True) or ascending order (False, default).
# Example:

numbers = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
numbers.sort()
print(numbers)  # Output: [1, 1, 2, 3, 3, 4, 5, 5, 6, 9]


# Custom Sorting with Key Function:

# Both sorted() and sort() functions accept a key parameter, which specifies a function to be applied to each element for sorting purposes.
# The key function determines the sorting order by transforming each element before comparison.
# Example:

names = ['John', 'Alice', 'Bob', 'Charlie']
sorted_names = sorted(names, key=len)  # Sort by length of names
print(sorted_names)  # Output: ['Bob', 'John', 'Alice', 'Charlie']


# Reverse Sorting:

# Both sorted() and sort() functions accept a reverse parameter, which specifies whether to sort in descending order (reverse=True) or ascending order (default).
# Example: sorted(numbers, reverse=True) or numbers.sort(reverse=True)

# Sorting lists is a fundamental operation in Python programming, and understanding these methods allows you to organize data efficiently based on 
# specific criteria. Additionally, custom sorting with key functions provides flexibility for sorting complex data structures.