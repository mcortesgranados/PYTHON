import numpy as np

# Example: NumPy Array Creation

# 1. Creating arrays from Python lists
arr1 = np.array([1, 2, 3, 4, 5])
print("Array from Python list:")
print(arr1)

# 2. Creating arrays of zeros
arr2 = np.zeros((3, 4))  # Create a 3x4 array of zeros
print("\nArray of zeros:")
print(arr2)

# 3. Creating arrays of ones
arr3 = np.ones((2, 3))  # Create a 2x3 array of ones
print("\nArray of ones:")
print(arr3)

# 4. Creating arrays of constant values
arr4 = np.full((3, 3), 5)  # Create a 3x3 array filled with 5s
print("\nArray of constant values:")
print(arr4)

# 5. Creating arrays with a range of values
arr5 = np.arange(0, 10, 2)  # Create an array from 0 to 10 (exclusive) with step size 2
print("\nArray with a range of values:")
print(arr5)

# 6. Creating arrays with evenly spaced values
arr6 = np.linspace(0, 1, 5)  # Create an array of 5 evenly spaced values between 0 and 1
print("\nArray with evenly spaced values:")
print(arr6)

# 7. Creating identity matrices
arr7 = np.eye(3)  # Create a 3x3 identity matrix
print("\nIdentity matrix:")
print(arr7)

# 8. Creating diagonal matrices
arr8 = np.diag([1, 2, 3, 4])  # Create a diagonal matrix with diagonal elements [1, 2, 3, 4]
print("\nDiagonal matrix:")
print(arr8)

# 9. Creating random arrays
arr9 = np.random.rand(3, 3)  # Create a 3x3 array of random numbers between 0 and 1
print("\nRandom array:")
print(arr9)

# 10. Creating arrays with specific data types
arr10 = np.array([1, 2, 3], dtype=np.float32)  # Create an array with float32 data type
print("\nArray with specific data type:")
print(arr10)

# Documenting the NumPy Array Creation:
def numpy_array_creation_documentation():
    """
    This function demonstrates various ways of creating NumPy arrays.

    Examples:
    - Creating arrays from Python lists using numpy.array().
    - Creating arrays of zeros using numpy.zeros().
    - Creating arrays of ones using numpy.ones().
    - Creating arrays of constant values using numpy.full().
    - Creating arrays with a range of values using numpy.arange().
    - Creating arrays with evenly spaced values using numpy.linspace().
    - Creating identity matrices using numpy.eye().
    - Creating diagonal matrices using numpy.diag().
    - Creating random arrays using numpy.random.rand().
    - Creating arrays with specific data types using the dtype parameter.
    """
    pass

# End of example
