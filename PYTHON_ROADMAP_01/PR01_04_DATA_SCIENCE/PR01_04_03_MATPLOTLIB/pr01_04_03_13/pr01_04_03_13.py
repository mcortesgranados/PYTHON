"""
13. Generating scatter plots with different marker sizes and colors for each data point.

"""

import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(0)
x = np.random.rand(50)
y = np.random.rand(50)
sizes = np.random.rand(50) * 100  # Random sizes for markers
colors = np.random.rand(50)       # Random colors for markers

# Create scatter plot
plt.scatter(x, y, s=sizes, c=colors, alpha=0.5, cmap='viridis')

# Add color bar to indicate size scale
plt.colorbar(label='Marker Size')

# Add labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot with Marker Sizes and Colors')

# Show plot
plt.show()
