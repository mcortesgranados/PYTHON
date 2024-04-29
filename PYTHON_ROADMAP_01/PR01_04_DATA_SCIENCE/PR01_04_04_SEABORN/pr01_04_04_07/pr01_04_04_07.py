"""
07. Building strip plots to visualize individual data points along with a scatter plot-like representation.

"""

import seaborn as sns
import matplotlib.pyplot as plt

# Sample data
data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5]

# Create a strip plot
sns.stripplot(data, jitter=True)

# Add labels and title
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Strip Plot Example')

# Show plot
plt.show()
