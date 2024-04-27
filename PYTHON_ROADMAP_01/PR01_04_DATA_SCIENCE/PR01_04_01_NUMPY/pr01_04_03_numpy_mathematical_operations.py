import numpy as np

# Example: Mathematical Operations with NumPy

# Create sample arrays
arr1 = np.array([[1, 2, 3],
                  [4, 5, 6]])
arr2 = np.array([[7, 8, 9],
                  [10, 11, 12]])

# Element-wise operations
# Addition
arr_sum = arr1 + arr2
print("Array Sum (Element-wise addition):")
print(arr_sum)

# Subtraction
arr_diff = arr1 - arr2
print("\nArray Difference (Element-wise subtraction):")
print(arr_diff)

# Multiplication
arr_prod = arr1 * arr2
print("\nArray Product (Element-wise multiplication):")
print(arr_prod)

# Division
arr_div = arr2 / arr1
print("\nArray Division (Element-wise division):")
print(arr_div)

# Trigonometric functions
arr_sin = np.sin(arr1)
print("\nSine of Array (Element-wise sine):")
print(arr_sin)

arr_cos = np.cos(arr1)
print("\nCosine of Array (Element-wise cosine):")
print(arr_cos)

arr_tan = np.tan(arr1)
print("\nTangent of Array (Element-wise tangent):")
print(arr_tan)

# Exponential and logarithmic functions
arr_exp = np.exp(arr1)
print("\nExponential of Array (Element-wise exponential):")
print(arr_exp)

arr_log = np.log(arr1)
print("\nLogarithm of Array (Element-wise natural logarithm):")
print(arr_log)

# Documenting the Mathematical Operations with NumPy:
def numpy_math_operations_documentation():
    """
    This function demonstrates various mathematical operations that can be performed on NumPy arrays.

    Examples:
    - Element-wise operations like addition, subtraction, multiplication, and division.
    - Trigonometric functions like numpy.sin(), numpy.cos(), numpy.tan().
    - Exponential and logarithmic functions like numpy.exp(), numpy.log().
    """
    pass

# End of example
