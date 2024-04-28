"""
25. Plotting waterfall charts to visualize cumulative effect of sequentially introduced positive or negative values.

"""

import matplotlib.pyplot as plt

# Data
categories = ['Start', 'Step 1', 'Step 2', 'Step 3', 'End']
values = [100, -20, 30, -10, 120]

# Calculate cumulative values
cumulative_values = [sum(values[:i+1]) for i in range(len(values))]

# Create waterfall chart
plt.bar(categories, cumulative_values, color='skyblue')

# Add labels and title
plt.xlabel('Categories')
plt.ylabel('Cumulative Values')
plt.title('Waterfall Chart Example')

# Show plot
plt.grid(axis='y')
plt.show()
