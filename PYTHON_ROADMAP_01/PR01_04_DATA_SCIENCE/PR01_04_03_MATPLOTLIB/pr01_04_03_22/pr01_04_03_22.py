"""
22. Plotting stream plots to visualize 2D vector fields using streamlines.

"""

import numpy as np
import matplotlib.pyplot as plt

# Define grid
Y, X = np.mgrid[-3:3:100j, -3:3:100j]

# Define vector field
U = -1 - X**2 + Y
V = 1 + X - Y**2

# Create stream plot
plt.streamplot(X, Y, U, V, color='blue')

# Add labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Stream Plot Example')

# Show plot
plt.show()
