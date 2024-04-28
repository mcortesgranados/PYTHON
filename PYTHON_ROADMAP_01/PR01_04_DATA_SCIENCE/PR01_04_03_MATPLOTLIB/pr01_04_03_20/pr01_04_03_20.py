"""
20. Generating violin plots to visualize the distribution of data like box plots.

"""

import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
data = [np.random.normal(0, std, 100) for std in range(1, 4)]

# Create violin plot
plt.figure(figsize=(8, 6))
plt.violinplot(data, showmeans=False, showmedians=True)

# Add labels and title
plt.xlabel('Data')
plt.ylabel('Value')
plt.title('Violin Plot Example')

# Show plot
plt.show()
