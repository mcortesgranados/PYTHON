import numpy as np

# Creating arrays
array1d = np.array([1, 2, 3, 4, 5])
print("1D Array:")
print(array1d)

array2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("\n2D Array:")
print(array2d)

# Array properties
print("\nShape of array2d:", array2d.shape)
print("Number of dimensions:", array2d.ndim)
print("Data type of elements:", array2d.dtype)
print("Size of array2d:", array2d.size)

# Accessing elements
print("\nFirst row:", array2d[0])
print("Second column:", array2d[:, 1])
print("Element at row 1, column 2:", array2d[0, 1])

# Array operations
array1 = np.array([1, 2, 3])
array2 = np.array([4, 5, 6])

print("\nArray addition:", array1 + array2)
print("Array subtraction:", array1 - array2)
print("Array multiplication:", array1 * array2)
print("Array division:", array1 / array2)

# Universal functions (ufunc)
print("\nSin values of array1:", np.sin(array1))
print("Exponential values of array2:", np.exp(array2))
print("Sum of all elements in array1:", np.sum(array1))
print("Mean of all elements in array2:", np.mean(array2))
