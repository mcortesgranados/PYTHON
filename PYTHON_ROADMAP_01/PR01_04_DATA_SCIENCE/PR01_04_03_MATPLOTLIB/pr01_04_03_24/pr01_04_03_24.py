"""
24. Creating annotated plots with text annotations and arrows.

"""

import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create plot
plt.plot(x, y)

# Add text annotation
plt.text(3, 0, 'Text Annotation', fontsize=12, color='blue')

# Add arrow annotation
plt.annotate('Arrow Annotation', xy=(np.pi, 0), xytext=(np.pi + 1, 0.5),
             arrowprops=dict(facecolor='red', shrink=0.05))

# Add labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Annotated Plot Example')

# Show plot
plt.grid(True)
plt.show()
