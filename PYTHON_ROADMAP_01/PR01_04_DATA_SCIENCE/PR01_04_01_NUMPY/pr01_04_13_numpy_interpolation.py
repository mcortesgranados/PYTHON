# python .\PYTHON_ROADMAP_01\PR01_04_DATA_SCIENCE\PR01_04_01_NUMPY\pr01_04_13_interpolation.py

import numpy as np
import matplotlib.pyplot as plt

# Sample data points
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 3, 1, 5, 7])

# Interpolate data using numpy.interp()
x_new = np.linspace(1, 5, 10)  # Generate new x values for interpolation
y_interp = np.interp(x_new, x, y)  # Interpolate y values for the new x values

# Plot the original data and interpolated data
plt.figure(figsize=(8, 6))
plt.plot(x, y, 'o', label='Original Data')
plt.plot(x_new, y_interp, '--', label='Interpolated Data')
plt.title('Interpolation using numpy.interp()')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
