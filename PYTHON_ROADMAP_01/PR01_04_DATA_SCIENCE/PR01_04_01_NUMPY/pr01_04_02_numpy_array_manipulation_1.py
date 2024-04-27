import numpy as np

# Example: Array Manipulation with NumPy

# Create a sample array
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

print("Original Array:")
print(arr)

# 1. Reshaping arrays using numpy.reshape()
arr_reshaped = np.reshape(arr, (1, 9))  # Reshape the array to a 1x9 array
print("\nReshaped Array:")
print(arr_reshaped)

# 2. Flattening arrays using numpy.flatten() or numpy.ravel()
arr_flattened = arr.flatten()  # Flatten the array to a 1D array
print("\nFlattened Array:")
print(arr_flattened)

# 3. Transposing arrays using numpy.transpose() or array.T
arr_transposed = np.transpose(arr)  # Transpose the array
print("\nTransposed Array:")
print(arr_transposed)

# 4. Concatenating arrays using numpy.concatenate()
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])
arr_concatenated = np.concatenate((arr1, arr2), axis=0)  # Concatenate arrays along the rows
print("\nConcatenated Array along rows:")
print(arr_concatenated)

# 5. Splitting arrays using numpy.split()
arr_split = np.split(arr_concatenated, 2, axis=0)  # Split the array into 2 equal parts along the rows
print("\nSplit Arrays:")
print(arr_split)

# 6. Stacking arrays using numpy.stack()
arr_stacked = np.stack((arr1, arr2), axis=0)  # Stack arrays along a new axis (axis=0)
print("\nStacked Arrays along new axis:")
print(arr_stacked)

# 7. Adding new dimensions to arrays using numpy.expand_dims()
arr_new_dimension = np.expand_dims(arr, axis=2)  # Add a new dimension to the array
print("\nArray with new dimension:")
print(arr_new_dimension)

# Documenting the Array Manipulation with NumPy:
def numpy_array_manipulation_documentation():
    """
    This function demonstrates various array manipulation techniques using NumPy.

    Examples:
    - Reshaping arrays using numpy.reshape().
    - Flattening arrays using numpy.flatten() or numpy.ravel().
    - Transposing arrays using numpy.transpose() or array.T.
    - Concatenating arrays using numpy.concatenate().
    - Splitting arrays using numpy.split().
    - Stacking arrays using numpy.stack().
    - Adding new dimensions to arrays using numpy.expand_dims().
    """
    pass

# End of example
