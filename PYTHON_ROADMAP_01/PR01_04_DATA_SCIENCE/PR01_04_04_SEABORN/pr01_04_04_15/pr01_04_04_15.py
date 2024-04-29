"""
15. Building factor plots to visualize categorical variables across one or more factors.

"""

import seaborn as sns
import matplotlib.pyplot as plt

# Load a sample dataset from Seaborn
tips = sns.load_dataset("tips")

# Create a factor plot
sns.factorplot(x="day", y="total_bill", hue="sex", data=tips, kind="bar", palette="Set1")

# Set plot title and labels
plt.title('Factor Plot Example')
plt.xlabel('Day of the Week')
plt.ylabel('Total Bill ($)')

# Show plot
plt.show()
