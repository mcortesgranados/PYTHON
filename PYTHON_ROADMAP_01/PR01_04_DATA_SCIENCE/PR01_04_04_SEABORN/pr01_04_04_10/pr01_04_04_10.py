"""
010. Creating joint plots to visualize the joint distribution between two variables along with their marginal distributions.

"""

import seaborn as sns
import matplotlib.pyplot as plt

# Sample data
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# Create a joint plot
sns.jointplot(x=x, y=y, kind='scatter')

# Show plot
plt.show()


