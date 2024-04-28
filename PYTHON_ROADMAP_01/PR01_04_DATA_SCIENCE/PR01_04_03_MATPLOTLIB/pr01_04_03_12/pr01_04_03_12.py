"""
12. Plotting polar plots to represent data in polar coordinates.

"""

import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
categories = ['Category A', 'Category B', 'Category C', 'Category D', 'Category E']
values = [4, 3, 2, 5, 4]

# Number of categories
num_categories = len(categories)

# Compute angle for each category (equally spaced)
angles = np.linspace(0, 2 * np.pi, num_categories, endpoint=False).tolist()

# Repeat the first data point to close the circle
values += values[:1]
angles += angles[:1]

# Create polar plot
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.fill(angles, values, color='skyblue', alpha=0.5)
ax.plot(angles, values, color='blue', linewidth=2)

# Add labels
ax.set_yticklabels([])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)

# Add title
plt.title('Polar Plot Example')

# Show plot
plt.show()
