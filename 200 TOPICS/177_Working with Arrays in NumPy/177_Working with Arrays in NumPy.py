# In this script:

# We create NumPy arrays array1 and array2.
# We print the arrays to display their contents.
# We access individual elements of the arrays using indexing.
# We slice the arrays to extract specific portions of the data.
# We reshape one array (array3) into a 3x3 matrix.
# We perform basic array operations such as sum and mean.
# We perform element-wise addition of two arrays using the np.add function.
# Each section of the script is accompanied by explanatory comments to help understand the purpose of the code.
#  You can run this script to see the output and better understand how to work with arrays in NumPy.

import numpy as np

# Creating NumPy arrays
array1 = np.array([1, 2, 3, 4, 5])  # 1D array
array2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # 2D array

# Printing arrays
print("Array 1:")  # Print array 1
print(array1)  # Print array 1
print("\nArray 2:")  # Print array 2
print(array2)  # Print array 2

# Accessing elements
print("\nElement at index 2 in array 1:", array1[2])  # Print element at index 2 in array 1
print("Element at row 1, column 2 in array 2:", array2[1, 2])  # Print element at row 1, column 2 in array 2

# Slicing arrays
print("\nSlice of array 1 from index 1 to 3:", array1[1:4])  # Print slice of array 1 from index 1 to 3
print("Slice of array 2 from row 0 to 2, column 1:", array2[:3, 1])  # Print slice of array 2 from row 0 to 2, column 1

# Reshaping arrays
array3 = np.arange(9).reshape((3, 3))  # Create a 3x3 array
print("\nArray 3:")  # Print array 3
print(array3)  # Print array 3

# Array operations
print("\nSum of array 1:", np.sum(array1))  # Print sum of array 1
print("Mean of array 2:", np.mean(array2))  # Print mean of array 2
print("Element-wise addition of array 1 and array 2:")  # Print element-wise addition of array 1 and array 2
print(np.add(array1, array2))  # Print result of element-wise addition of array 1 and array 2
