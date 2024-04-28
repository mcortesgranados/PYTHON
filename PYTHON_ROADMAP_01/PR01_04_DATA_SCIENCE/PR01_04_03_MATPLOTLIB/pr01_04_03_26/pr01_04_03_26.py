"""
26. Generating barh plots to create horizontal bar plots.

"""

import matplotlib.pyplot as plt

# Data
categories = ['Category A', 'Category B', 'Category C', 'Category D']
values = [20, 35, 30, 25]

# Create horizontal bar plot
plt.barh(categories, values, color='skyblue')

# Add labels and title
plt.xlabel('Values')
plt.ylabel('Categories')
plt.title('Horizontal Bar Plot Example')

# Show plot
plt.grid(axis='x')
plt.show()
