"""
22. Plotting pairgrid to visualize pairwise relationships in a dataset for multiple variables.

Documentation:

seaborn.PairGrid: https://seaborn.pydata.org/generated/seaborn.pairplot.html
pandas.DataFrame: https://pandas.pydata.org/docs/getting_started/intro_tutorials/
seaborn.scatterplot: https://seaborn.pydata.org/generated/seaborn.scatterplot.html
seaborn.histplot: https://seaborn.pydata.org/generated/seaborn.histplot.html

"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Create sample data (assuming numerical data)
data = {'x1': [1, 3, 4, 2, 5],
        'x2': [6, 5, 8, 1, 4],
        'x3': [7, 2, 9, 4, 1]}
df = pd.DataFrame(data)

# Create a PairGrid using Seaborn
sns.set_style("whitegrid")
g = sns.PairGrid(df)

# Map scatter plots to the upper triangle
g.map_upper(sns.scatterplot)  # Can also use other plot types like 'regplot'

# Map histograms to the diagonal
g.map_diag(sns.histplot)

# Adjust the plot layout (optional)
g.fig.suptitle('Pairwise Relationships between Variables')

# Display the plot
plt.show()
