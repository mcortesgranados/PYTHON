"""
02. Generating scatter plots to explore the correlation between two continuous variables.

"""

import seaborn as sns
import matplotlib.pyplot as plt

# Sample data
x_values = [1, 2, 3, 4, 5]
y_values = [2, 4, 6, 8, 10]

# Create a scatter plot
sns.scatterplot(x=x_values, y=y_values)

# Add labels and title
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')
plt.title('Scatter Plot Example')

# Show plot
plt.show()
