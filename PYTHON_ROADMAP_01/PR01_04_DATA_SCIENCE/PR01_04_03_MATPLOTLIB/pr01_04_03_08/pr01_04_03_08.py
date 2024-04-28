"""
08. Plotting error bars to visualize uncertainty or variability in data points.

"""

import matplotlib.pyplot as plt

# Sample data
x = [1, 2, 3, 4, 5]
y = [10, 15, 20, 25, 30]
error = [1, 2, 1.5, 2.5, 1]

# Create a plot with error bars
plt.errorbar(x, y, yerr=error, fmt='-o', ecolor='red', capsize=5)

# Add labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Error Bar Plot Example')

# Show plot
plt.show()

