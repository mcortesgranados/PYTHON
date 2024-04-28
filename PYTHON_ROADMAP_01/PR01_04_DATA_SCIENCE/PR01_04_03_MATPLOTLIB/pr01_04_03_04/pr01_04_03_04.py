"""
04. Plotting histograms to display the distribution of a single variable.
"""

import matplotlib.pyplot as plt

# Sample data
data = [1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5]

# Create a histogram
plt.hist(data, bins=5, color='salmon', edgecolor='black')

# Add labels and title
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Histogram Example')

# Show plot
plt.show()
