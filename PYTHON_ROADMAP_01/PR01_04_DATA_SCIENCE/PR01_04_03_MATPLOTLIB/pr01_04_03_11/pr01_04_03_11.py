"""
11. Building quiver plots to visualize vector fields using arrows.

"""
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data (vector field)
x = np.linspace(-2, 2, 10)
y = np.linspace(-2, 2, 10)
X, Y = np.meshgrid(x, y)
U = np.cos(X)
V = np.sin(Y)

# Create a quiver plot
plt.quiver(X, Y, U, V)

# Add labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Quiver Plot Example')

# Show plot
plt.show()
