"""
16. Plotting count plots to display the count of observations in each category of a categorical variable.

"""

import seaborn as sns
import matplotlib.pyplot as plt

# Load a sample dataset from Seaborn
tips = sns.load_dataset("tips")

# Create a count plot
sns.countplot(x="day", data=tips, palette="Set2")

# Set plot title and labels
plt.title('Count Plot Example')
plt.xlabel('Day of the Week')
plt.ylabel('Count')

# Show plot
plt.show()
