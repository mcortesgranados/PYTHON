"""
Polynomial fitting is a technique used to find a polynomial function that best fits a set of data points. 
The numpy.polyfit() function is commonly used for this purpose. Here's an example demonstrating polynomial fitting using NumPy:

python .\PYTHON_ROADMAP_01\PR01_04_DATA_SCIENCE\PR01_04_01_NUMPY\pr01_04_18_numpy_polynomial_fitting.py

"""

import numpy as np
import matplotlib.pyplot as plt

# Generate some sample data points
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([1, 3, 2, 5, 4, 6])

# Perform polynomial fitting
degree = 2  # Degree of the polynomial
coefficients = np.polyfit(x, y, degree)

# Create the polynomial function
poly_function = np.poly1d(coefficients)

# Generate points for the polynomial curve
x_curve = np.linspace(0, 5, 100)
y_curve = poly_function(x_curve)

# Plot the original data points and the polynomial curve
plt.figure(figsize=(8, 6))
plt.plot(x, y, 'o', label='Data Points')
plt.plot(x_curve, y_curve, label='Polynomial Fit (Degree {})'.format(degree))
plt.title('Polynomial Fitting using numpy.polyfit()')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
