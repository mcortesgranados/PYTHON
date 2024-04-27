"""

Explanation:

We import the NumPy library as np.
We create sample 1D and 2D arrays, arr_1d and arr_2d, respectively.
For 1D arrays:
We access elements using basic indexing (arr_1d[2]).
We slice the array to get subarrays (arr_1d[1:4]).
We perform boolean indexing to select elements based on a condition (arr_1d[mask]).
We use fancy indexing to select elements at specified indices (arr_1d[indices]).
For 2D arrays:
We access elements using basic indexing (arr_2d[1, 2]).
We slice the array to get subarrays (arr_2d[:2, 1:3]).
We perform boolean indexing to select elements based on a condition (arr_2d[mask_2d]).
We use fancy indexing to select elements at specified row and column indices (arr_2d[rows, cols]).

Documentation:

Indexing and slicing in NumPy arrays: https://numpy.org/doc/stable/user/basics.indexing.html
This example demonstrates various techniques for accessing elements and subarrays of NumPy arrays using indexing,
 slicing, boolean indexing, and fancy indexing in Python.

"""

import numpy as np

# Create a sample 1D array
arr_1d = np.array([0, 1, 2, 3, 4, 5])

# Accessing elements using basic indexing
print("Element at index 2:", arr_1d[2])

# Slicing to get subarrays
print("Subarray from index 1 to 3:", arr_1d[1:4])

# Boolean indexing
mask = arr_1d > 2
print("Elements greater than 2:", arr_1d[mask])

# Fancy indexing
indices = [0, 2, 4]
print("Elements at specified indices:", arr_1d[indices])


# Create a sample 2D array
arr_2d = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])

# Accessing elements using basic indexing
print("Element at row 1, column 2:", arr_2d[1, 2])

# Slicing to get subarrays
print("Subarray from rows 0 to 1, columns 1 to 2:")
print(arr_2d[:2, 1:3])

# Boolean indexing
mask_2d = arr_2d > 5
print("Elements greater than 5:")
print(arr_2d[mask_2d])

# Fancy indexing
rows = [0, 2]
cols = [1, 2]
print("Elements at specified row and column indices:")
print(arr_2d[rows, cols])
