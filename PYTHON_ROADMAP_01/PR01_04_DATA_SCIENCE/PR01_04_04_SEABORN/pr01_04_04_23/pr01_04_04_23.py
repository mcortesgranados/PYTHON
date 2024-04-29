"""
23. Generating FacetGrid to create a grid of subplots based on one or more categorical variables.

Documentation:

seaborn.FacetGrid: https://seaborn.pydata.org/generated/seaborn.FacetGrid.html
pandas.DataFrame: [invalid URL removed]
seaborn.scatterplot: https://seaborn.pydata.org/generated/seaborn.scatterplot.html

"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Create sample data
data = {'treatment': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
        'value': [2, 1, 4, 5, 3, 7, 1, 2, 4],
        'group': ['control', 'control', 'control', 'treated', 'treated', 'treated', 'treated', 'treated', 'treated']}
df = pd.DataFrame(data)

# Create a FacetGrid with Seaborn
sns.set_style("whitegrid")
g = sns.FacetGrid(df, col="treatment", row="group")  # col for column-wise split, row for row-wise

# Map a scatter plot to each subplot
g.map(sns.scatterplot, x="value", y="treatment")   # x and y can be adjusted based on the data

# Add a title and adjust layout (optional)
g.fig.suptitle('Comparison of Values by Treatment Group')
g.fig.subplots_adjust(top=0.9)  # Adjust spacing between subplots and title

# Display the plot
plt.show()


