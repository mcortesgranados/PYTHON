"""
17. Generating stem plots to visualize discrete data points.

"""
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
x = np.arange(10)
y = np.random.randint(1, 10, size=10)

# Create stem plot
plt.figure(figsize=(8, 6))
plt.stem(x, y, linefmt='b-', markerfmt='bo', basefmt='r-')

# Add labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Stem Plot Example')

# Show plot
plt.show()
