"""
23. Generating scatter plots with regression lines to visualize linear relationships.

Explanation:

We import the numpy module as np for numerical computations and the matplotlib.pyplot module as plt for creating plots.
We generate sample data for the x and y variables. Here, x is randomly generated, and y is calculated using a linear relationship with some added noise.
We fit a linear regression line to the data using np.polyfit() with degree 1, which returns the slope and intercept of the regression line.
We create a scatter plot of the data points using plt.scatter() with the x and y variables. We also specify a label for the data points.
We plot the regression line using plt.plot() with the x variable and the calculated regression line. We specify the color of the line as red and provide a label for the regression line.
We add labels to the x-axis and y-axis using plt.xlabel() and plt.ylabel(), respectively.
We set a title for the plot using plt.title().
We add a legend to the plot using plt.legend() to differentiate between the data points and the regression line.
Finally, we display the plot using plt.show().

"""

import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
x = np.random.rand(50) * 10
y = 2 * x + np.random.randn(50)

# Fit a linear regression line
slope, intercept = np.polyfit(x, y, 1)
regression_line = slope * x + intercept

# Create scatter plot with regression line
plt.scatter(x, y, label='Data Points')
plt.plot(x, regression_line, color='red', label='Regression Line')

# Add labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot with Regression Line')

# Add legend
plt.legend()

# Show plot
plt.grid(True)
plt.show()




