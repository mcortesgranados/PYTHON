"""
11. Building rug plots to visualize the distribution of data points along a single axis.

"""

import seaborn as sns
import matplotlib.pyplot as plt

# Sample data
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Create a rug plot
sns.rugplot(x=data, height=0.5)

# Set plot title and labels
plt.title('Rug Plot Example')
plt.xlabel('Data Points')
plt.ylabel('Density')

# Show plot
plt.show()
