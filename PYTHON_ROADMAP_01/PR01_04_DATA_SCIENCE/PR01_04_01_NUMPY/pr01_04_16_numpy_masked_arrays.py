"""
Masked arrays are useful for handling missing or invalid data in NumPy arrays. Here's an example demonstrating the concept:

"""

import numpy as np
import numpy.ma as ma

# Define a NumPy array with some missing values
data = np.array([1, 2, -999, 4, -999, 6, 7, -999, 9])

# Create a mask for the missing values
mask = data == -999

# Create a masked array
masked_data = ma.masked_array(data, mask=mask)

# Print the original array and the masked array
print("Original array:", data)
print("Masked array:", masked_data)

# Perform operations on the masked array
mean_value = ma.mean(masked_data)
sum_value = ma.sum(masked_data)

# Print the results
print("Mean value (ignoring missing values):", mean_value)
print("Sum value (ignoring missing values):", sum_value)
