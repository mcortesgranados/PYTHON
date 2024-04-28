"""
15. Plotting 3D surface plots to visualize functions of two variables in 3D space.

"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the function to plot (in this case, a saddle-shaped function)
def saddle(x, y):
    return x**2 - y**2

# Generate data points for x and y coordinates
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)  # Create a meshgrid of x and y coordinates

# Compute the corresponding z values using the defined function
Z = saddle(X, Y)

# Create a 3D plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(X, Y, Z, cmap='viridis')

# Add labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Surface Plot Example')

# Add color bar
fig.colorbar(surf, shrink=0.5, aspect=5)

# Show plot
plt.show()
