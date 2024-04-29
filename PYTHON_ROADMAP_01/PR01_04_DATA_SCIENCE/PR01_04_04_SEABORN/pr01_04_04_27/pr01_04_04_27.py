"""
27. Creating bar plots with hue for additional categorical grouping.

seaborn.barplot: https://seaborn.pydata.org/generated/seaborn.barplot.html
pandas.DataFrame: [invalid URL removed]

"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Create sample data
data = {'treatment': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
        'size': [20, 30, 15, 25, 40, 18, 35, 22, 12],
        'group': ['control', 'control', 'control', 'treated', 'treated', 'treated', 'treated', 'treated', 'treated']}
df = pd.DataFrame(data)

# Create a bar plot with hue for grouping
sns.set_style("whitegrid")
sns.barplot(x="treatment", y="size", hue="group", data=df)

# Add a title and axis labels (optional)
plt.title("Distribution of Sizes by Treatment Group")
plt.xlabel("Treatment")
plt.ylabel("Size")

# Display the plot
plt.show()
