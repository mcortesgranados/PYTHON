"""
01. Creating line plots to visualize trends or relationships between variables.

"""

import seaborn as sns
import matplotlib.pyplot as plt

# Sample data
x_values = [1, 2, 3, 4, 5]
y_values = [2, 4, 6, 8, 10]

# Create a line plot
sns.lineplot(x=x_values, y=y_values)

# Add labels and title
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')
plt.title('Line Plot Example')

# Show plot
plt.show()
