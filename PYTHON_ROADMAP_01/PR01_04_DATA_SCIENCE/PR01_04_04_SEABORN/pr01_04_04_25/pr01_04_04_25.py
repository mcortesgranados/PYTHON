"""
25. Plotting violin plots with hue for additional categorical grouping.

seaborn.violinplot (for Seaborn >= 0.11): https://seaborn.pydata.org/generated/seaborn.violinplot.html
pandas.DataFrame: https://pandas.pydata.org/docs/getting_started/intro_tutorials/

"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Create sample data
data = {'treatment': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
        'size': [20, 30, 15, 25, 40, 18, 35, 22, 12],
        'group': ['control', 'control', 'control', 'treated', 'treated', 'treated', 'treated', 'treated', 'treated']}
df = pd.DataFrame(data)

# Create a violin plot with hue for grouping (Seaborn version >= 0.11)
sns.set_style("whitegrid")
sns.violinplot(x="treatment", y="size", hue="group", data=df, split=True)  # split for separate violins per hue

# Add a title and axis labels (optional)
plt.title("Distribution of Sizes by Treatment Group")
plt.xlabel("Treatment")
plt.ylabel("Size")

# Display the plot
plt.show()
