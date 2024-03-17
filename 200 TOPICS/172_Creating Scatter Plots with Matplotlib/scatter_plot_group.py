# scatter_plot_group.py

import matplotlib.pyplot as plt

# Data
x1 = [1, 2, 3, 4, 5]
y1 = [2, 3, 5, 7, 11]
x2 = [1.5, 2.5, 3.5, 4.5, 5.5]
y2 = [3, 4, 6, 8, 12]

# Create a scatter plot with grouped data
plt.scatter(x1, y1, color='r', marker='o', label='Group 1')
plt.scatter(x2, y2, color='g', marker='s', label='Group 2')

# Customize the chart
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Scatter Plot with Grouped Data')
plt.legend()

# Display the chart
plt.show()
