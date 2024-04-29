"""
03. Building bar plots to compare categorical data or show frequency distributions.

"""

import seaborn as sns
import matplotlib.pyplot as plt

# Sample data
categories = ['A', 'B', 'C', 'D', 'E']
values = [10, 20, 15, 25, 30]

# Create a bar plot
sns.barplot(x=categories, y=values)

# Add labels and title
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Bar Plot Example')

# Show plot
plt.show()
