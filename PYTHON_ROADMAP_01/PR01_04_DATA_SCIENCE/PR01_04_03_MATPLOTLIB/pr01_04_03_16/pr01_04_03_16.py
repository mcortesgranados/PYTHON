"""
16. Building stacked bar plots to compare the proportion of different categories.

"""

import numpy as np
import matplotlib.pyplot as plt

# Define sample data
categories = ['Category 1', 'Category 2', 'Category 3']
values1 = [20, 30, 25]  # Values for the first group
values2 = [15, 25, 30]  # Values for the second group

# Create stacked bar plot
plt.figure(figsize=(8, 6))
plt.bar(categories, values1, label='Group 1', color='blue')
plt.bar(categories, values2, bottom=values1, label='Group 2', color='orange')

# Add legend, labels, and title
plt.legend()
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Stacked Bar Plot Example')

# Show plot
plt.show()
