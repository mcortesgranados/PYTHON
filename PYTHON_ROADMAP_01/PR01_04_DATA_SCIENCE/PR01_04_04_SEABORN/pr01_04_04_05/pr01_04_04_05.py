"""
05. Creating box plots to visualize the distribution of data and identify outliers.

"""

import seaborn as sns
import matplotlib.pyplot as plt

# Sample data
data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5]

# Create a box plot
sns.boxplot(data)

# Add labels and title
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Box Plot Example')

# Show plot
plt.show()
