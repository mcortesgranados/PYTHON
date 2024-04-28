"""
05. Creating box plots to visualize the distribution of data and identify outliers.
"""

import matplotlib.pyplot as plt

# Sample data
data = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]

# Create a box plot
plt.boxplot(data, vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue'))

# Add labels and title
plt.xlabel('Values')
plt.title('Box Plot Example')

# Show plot
plt.show()
