"""
28. Plotting box plots with hue for additional categorical grouping.
seaborn.boxplot: https://seaborn.pydata.org/generated/seaborn.boxplot.html


"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Create sample data
data = {'treatment': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
        'value': [2, 1, 4, 5, 3, 7, 1, 2, 4],
        'group': ['control', 'control', 'control', 'treated', 'treated', 'treated', 'treated', 'treated', 'treated']}
df = pd.DataFrame(data)

# Create a box plot with hue for grouping
sns.set_style("whitegrid")
sns.boxplot(x="treatment", y="value", hue="group", showmeans=True, data=df)  # showmeans for displaying means

# Add a title and axis labels (optional)
plt.title("Comparison of Values by Treatment Group")
plt.xlabel("Treatment")
plt.ylabel("Value")

# Display the plot
plt.show()
