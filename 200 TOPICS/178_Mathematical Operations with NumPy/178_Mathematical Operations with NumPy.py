import numpy as np

# Creating NumPy arrays
array1 = np.array([1, 2, 3, 4, 5])
array2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Printing arrays
print("Array 1:")
print(array1)
print("\nArray 2:")
print(array2)

# Basic mathematical operations
print("\nAddition:")
print(array1 + 2)  # Add scalar 2 to each element of array 1
print("\nSubtraction:")
print(array1 - 2)  # Subtract scalar 2 from each element of array 1
print("\nMultiplication:")
print(array1 * 2)  # Multiply each element of array 1 by scalar 2
print("\nDivision:")
print(array1 / 2)  # Divide each element of array 1 by scalar 2

# Element-wise mathematical operations
print("\nElement-wise addition:")
print(array1 + array2)  # Add corresponding elements of array 1 and array 2
print("\nElement-wise subtraction:")
print(array1 - array2)  # Subtract corresponding elements of array 1 and array 2
print("\nElement-wise multiplication:")
print(array1 * array2)  # Multiply corresponding elements of array 1 and array 2
print("\nElement-wise division:")
print(array1 / array2)  # Divide corresponding elements of array 1 by array 2

# Other mathematical operations
print("\nSquare root:")
print(np.sqrt(array1))  # Compute square root of each element of array 1
print("\nExponential:")
print(np.exp(array1))  # Compute exponential of each element of array 1
print("\nSine:")
print(np.sin(array1))  # Compute sine of each element of array 1
