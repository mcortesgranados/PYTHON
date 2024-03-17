# scatter_plot_customized.py

import matplotlib.pyplot as plt

# Data
x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]

# Create a scatter plot with customization
plt.scatter(x, y, color='skyblue', marker='o', s=100, alpha=0.7, label='Data')

# Customize the chart
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Customized Scatter Plot')
plt.legend()

# Display the chart
plt.show()
