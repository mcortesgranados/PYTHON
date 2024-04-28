"""
09. Generating heatmaps to represent the magnitude of values in a matrix using colors.

"""

import numpy as np
import matplotlib.pyplot as plt

# Sample data (2D array)
data = np.random.rand(10, 10)

# Create a heatmap
plt.imshow(data, cmap='hot', interpolation='nearest')

# Add color bar
plt.colorbar()

# Add labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Heatmap Example')

# Show plot
plt.show()
