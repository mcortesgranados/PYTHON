"""
10. Creating contour plots to display the 3D surface in 2D space with contour lines.

"""

import numpy as np
import matplotlib.pyplot as plt

# Generate sample data (3D surface)
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(X) + np.cos(Y)

# Create a contour plot
plt.contour(X, Y, Z, cmap='viridis')

# Add color bar
plt.colorbar()

# Add labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Contour Plot Example')

# Show plot
plt.show()
