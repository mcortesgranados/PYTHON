"""
13. Generating heatmaps to represent the magnitude of values in a matrix using colors.

"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Sample data: a 5x5 matrix
data = np.random.rand(5, 5)

# Create a heatmap
sns.heatmap(data, annot=True, cmap='viridis')

# Set plot title
plt.title('Heatmap Example')

# Show plot
plt.show()
