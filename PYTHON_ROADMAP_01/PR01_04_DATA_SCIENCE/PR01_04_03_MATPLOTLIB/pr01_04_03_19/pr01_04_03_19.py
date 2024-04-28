"""
19. Plotting hexbin plots to represent the density of points in hexagonal bins.

"""

import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
x = np.random.randn(1000)  # Random x-coordinates
y = np.random.randn(1000)  # Random y-coordinates

# Create hexbin plot
plt.figure(figsize=(8, 6))
plt.hexbin(x, y, gridsize=30, cmap='Blues')

# Add labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Hexbin Plot Example')

# Add colorbar
plt.colorbar(label='Density')

# Show plot
plt.show()
