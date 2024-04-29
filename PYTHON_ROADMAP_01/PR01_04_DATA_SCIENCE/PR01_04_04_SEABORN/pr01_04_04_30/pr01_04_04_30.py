"""
30. Creating catplot to combine various categorical plots into a single figure.

seaborn.catplot: https://seaborn.pydata.org/generated/seaborn.catplot.html

python .\PYTHON_ROADMAP_01\PR01_04_DATA_SCIENCE\PR01_04_04_SEABORN\pr01_04_04_30\pr01_04_04_30.py


"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Create sample data (categorical data)
data = {'treatment': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
        'size': [20, 30, 15, 25, 40, 18, 35, 22, 12],
        'group': ['control', 'control', 'control', 'treated', 'treated', 'treated', 'treated', 'treated', 'treated']}
df = pd.DataFrame(data)

# Create a CatPlot with violin plots and box plots
sns.set_style("whitegrid")
g = sns.catplot(x="treatment", 
                y="size", 
                hue="group", 
                col="group", 
                kind="violin", 
                data=df, 
                sharex=False)  # sharex for independent x-axes per column

# Add a title and rotate x-axis labels (optional)
g.fig.suptitle("Comparison of Size Distribution by Treatment and Group")
g.fig.subplots_adjust(bottom=0.2)  # Adjust space for rotated x-axis labels
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

# Display the plot
plt.show()
