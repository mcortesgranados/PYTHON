"""
01. Creating line plots to visualize trends or relationships between variables.

"""

import matplotlib.pyplot as plt
import numpy as np

# Sample data (x-axis and y-axis values)
x = np.linspace(0.0, 5.0, 100)  # Creates 100 points between 0 and 5
y = np.sin(x)  # Creates sine wave data points for x values

# Create the line plot
plt.plot(x, y, label='Sine Wave')  # Plots x vs y with label 'Sine Wave'

# Add labels and title for clarity
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Line Plot - Visualizing a Sine Wave')

# Customize the plot (optional)
plt.grid(True)  # Add gridlines for better readability
plt.legend()  # Show the label for the line

# Display the plot
plt.show()
