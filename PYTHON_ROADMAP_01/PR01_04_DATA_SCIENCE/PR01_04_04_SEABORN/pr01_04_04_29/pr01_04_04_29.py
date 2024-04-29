"""
29. Generating count plots with hue for additional categorical grouping.

"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Create sample data (categorical data)
data = {'category': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
        'value': ['x', 'y', 'x', 'y', 'x', 'y', 'x', 'y', 'x'],
        'group': ['control', 'control', 'control', 'treated', 'treated', 'treated', 'treated', 'treated', 'treated']}
df = pd.DataFrame(data)

# Create a count plot with hue for grouping
sns.set_style("whitegrid")
sns.countplot(x="category", hue="group", data=df)

# Add a title and axis labels (optional)
plt.title("Count of Values by Category and Group")
plt.xlabel("Category")
plt.ylabel("Count")

# Display the plot
plt.show()
