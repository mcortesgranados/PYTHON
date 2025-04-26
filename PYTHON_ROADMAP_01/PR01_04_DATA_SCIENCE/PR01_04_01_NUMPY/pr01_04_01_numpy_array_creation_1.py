import numpy as np

""""
📦 import numpy as np — You bring in NumPy and give it the nickname np to make typing faster.

📋 np.array() — Turns a regular Python list into a NumPy array, which is better for calculations.

⚪ np.zeros() — Creates an array full of zeros, perfect for starting from scratch.

➕ np.ones() — Makes an array filled with ones, ready to use.

🔢 np.full() — Fills an array with any number you want (like a 2x2 array full of 7s).

📈 np.arange() — Makes an array with numbers in a range, like 0, 2, 4, 6... easy sequences.

🛤️ np.linspace() — Creates an array with numbers evenly spaced between two points, super smooth.

🚀 Overall, these tools make it super quick and easy to build arrays for anything math or data related!

""""

# Example: Array Creation with NumPy

# 1. Creating arrays from Python lists using numpy.array()
arr_from_list = np.array([1, 2, 3, 4, 5])
print("Array from Python list:")
print(arr_from_list)

# 2. Creating arrays of zeros using numpy.zeros()
arr_zeros = np.zeros((2, 3))  # Create a 2x3 array of zeros
print("\nArray of zeros:")
print(arr_zeros)

# 3. Creating arrays of ones using numpy.ones()
arr_ones = np.ones((3, 2))  # Create a 3x2 array of ones
print("\nArray of ones:")
print(arr_ones)

# 4. Creating arrays of constant values using numpy.full()
arr_full = np.full((2, 2), 7)  # Create a 2x2 array filled with 7s
print("\nArray of constant values:")
print(arr_full)

# 5. Creating arrays with a range of values using numpy.arange()
arr_range = np.arange(0, 10, 2)  # Create an array from 0 to 10 (exclusive) with step size 2
print("\nArray with a range of values:")
print(arr_range)

# 6. Creating arrays with evenly spaced values using numpy.linspace()
arr_linspace = np.linspace(0, 1, 5)  # Create an array of 5 evenly spaced values between 0 and 1
print("\nArray with evenly spaced values:")
print(arr_linspace)

# Documenting the Array Creation with NumPy:
def numpy_array_creation_documentation():
    """
    This function demonstrates various ways of creating arrays with NumPy.

    Examples:
    - Creating arrays from Python lists using numpy.array().
    - Creating arrays of zeros using numpy.zeros().
    - Creating arrays of ones using numpy.ones().
    - Creating arrays of constant values using numpy.full().
    - Creating arrays with a range of values using numpy.arange().
    - Creating arrays with evenly spaced values using numpy.linspace().
    """
    pass

# End of example
