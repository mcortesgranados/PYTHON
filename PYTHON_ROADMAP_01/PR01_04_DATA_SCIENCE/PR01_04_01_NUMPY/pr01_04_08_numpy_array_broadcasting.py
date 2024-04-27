"""
Explanation:

We import the NumPy library as np.
We create a 1D NumPy array arr_1d containing [1, 2, 3].
We create a scalar scalar with the value 2.
By using the * operator between the array and the scalar, NumPy automatically performs broadcasting, 
where the scalar value is broadcasted to match the shape of the array.
The result of broadcasting is the element-wise multiplication of each element in the array by the scalar value.
The resulting array is [2, 4, 6].
Documentation:

NumPy broadcasting documentation: https://numpy.org/doc/stable/user/basics.broadcasting.html
This example demonstrates how NumPy performs array broadcasting, allowing operations between arrays of different shapes by automatically 
aligning dimensions. In this case, the scalar value is broadcasted to match the shape of the array for element-wise multiplication.

"""

import numpy as np

# Create a 1D array
arr_1d = np.array([1, 2, 3])

# Create a scalar
scalar = 2

# Perform broadcasting: multiplying array by a scalar
result = arr_1d * scalar

# Display the result
print("Result of broadcasting:", result)
