"""
02. Generating scatter plots to explore the correlation between two continuous variables.


"""

import matplotlib.pyplot as plt

# Sample data
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# Create a scatter plot
plt.scatter(x, y, color='b', marker='o', label='Scatter Plot')

# Add labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Scatter Plot Example')

# Add grid
plt.grid(True)

# Add legend
plt.legend()

# Show plot
plt.show()
