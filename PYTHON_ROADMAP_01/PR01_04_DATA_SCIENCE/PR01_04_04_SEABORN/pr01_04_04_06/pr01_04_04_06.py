"""
06. Generating violin plots to visualize the distribution of data similar to box plots but with a kernel density plot.

"""

import seaborn as sns
import matplotlib.pyplot as plt

# Sample data
data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5]

# Create a violin plot
sns.violinplot(data)

# Add labels and title
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Violin Plot Example')

# Show plot
plt.show()
