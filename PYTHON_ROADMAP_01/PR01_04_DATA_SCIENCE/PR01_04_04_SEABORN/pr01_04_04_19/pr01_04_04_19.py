"""
19. Plotting regression plots to visualize the relationship between two continuous variables along with a regression line.

"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Create sample data
data = {'x': [1, 2, 3, 4, 5],
        'y': [3, 5, 7, 2, 8]}
df = pd.DataFrame(data)

# Create a regression plot using Seaborn's lmplot
sns.set_style("whitegrid")
g = sns.lmplot(x="x", y="y", data=df)  # lmplot for linear model plot

# Customize the plot (optional)
g.set_title("Regression Plot - Relationship between x and y")
g.set_xlabel("X-axis Variable")
g.set_ylabel("Y-axis Variable")
plt.show()
