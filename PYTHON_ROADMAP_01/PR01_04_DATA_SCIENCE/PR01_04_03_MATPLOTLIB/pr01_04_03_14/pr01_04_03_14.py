"""
14. Creating filled plots to represent the area between two curves.

"""

import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Create filled plot
plt.fill_between(x, y1, y2, color='skyblue', alpha=0.5)

# Plot the curves
plt.plot(x, y1, label='sin(x)', color='blue')
plt.plot(x, y2, label='cos(x)', color='green')

# Add legend
plt.legend()

# Add labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Filled Plot Example')

# Show plot
plt.show()
