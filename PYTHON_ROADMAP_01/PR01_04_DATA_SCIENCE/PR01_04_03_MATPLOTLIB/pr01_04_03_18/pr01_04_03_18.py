"""
18. Creating step plots to visualize stepwise changes in data.

"""

import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
x = np.linspace(0, 10, 11)  # Create 11 data points from 0 to 10
y = np.random.randint(0, 10, size=11)  # Generate random y-values

# Create step plot
plt.figure(figsize=(8, 6))
plt.step(x, y, where='mid', label='Step Plot', color='blue')

# Add labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Step Plot Example')
plt.legend()

# Show plot
plt.grid(True)
plt.show()
