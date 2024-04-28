"""
27. Creating spider plots (radar charts) to display multivariate data on a two-dimensional chart of three or more quantitative variables.

"""

import numpy as np
import matplotlib.pyplot as plt

# Data
categories = ['Category 1', 'Category 2', 'Category 3', 'Category 4', 'Category 5']
values = [4, 3, 2, 5, 4]

# Number of variables
num_vars = len(categories)

# Compute angle for each category
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

# The plot is circular, so we "complete the loop" and append the start
values += values[:1]
angles += angles[:1]

# Create figure and axes
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

# Plot data
ax.fill(angles, values, color='skyblue', alpha=0.4)
ax.plot(angles, values, color='blue', linewidth=2, linestyle='solid')

# Add labels and title
ax.set_yticklabels([])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
plt.title('Spider Plot Example')

# Show plot
plt.show()
