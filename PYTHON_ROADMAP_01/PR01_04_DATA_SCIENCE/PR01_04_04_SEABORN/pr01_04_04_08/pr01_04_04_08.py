"""
08. Plotting swarm plots to visualize individual data points without overlapping.

"""

import seaborn as sns
import matplotlib.pyplot as plt

# Sample data
data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5]

# Create a swarm plot
sns.swarmplot(data)

# Add labels and title
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Swarm Plot Example')

# Show plot
plt.show()
