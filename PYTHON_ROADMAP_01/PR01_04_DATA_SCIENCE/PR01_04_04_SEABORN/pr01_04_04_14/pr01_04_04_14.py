"""
14. Creating clustermaps to visualize hierarchical clustering of variables and observations.

"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Generate sample data: a 10x10 matrix
data = np.random.rand(10, 10)

# Create a clustermap
sns.clustermap(data, cmap='viridis')

# Set plot title
plt.title('Clustermap Example')

# Show plot
plt.show()
