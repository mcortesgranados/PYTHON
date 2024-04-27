import numpy as np

# Generate some sample data points
data = np.array([1.2, 2.5, 3.7, 4.1, 5.8])

# Define bins for quantization
bins = np.array([0, 2, 4, 6])

# Perform quantization using numpy.digitize()
quantized_data = np.digitize(data, bins)

# Print the original data and the quantized data
print("Original Data:", data)
print("Quantized Data:", quantized_data)
