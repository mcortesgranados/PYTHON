"""
12. Plotting KDE (Kernel Density Estimate) plots to estimate the probability density function of a continuous variable.

"""

import seaborn as sns
import matplotlib.pyplot as plt

# Sample data
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Create a KDE plot
sns.kdeplot(data)

# Set plot title and labels
plt.title('KDE Plot Example')
plt.xlabel('Data Points')
plt.ylabel('Density')

# Show plot
plt.show()
