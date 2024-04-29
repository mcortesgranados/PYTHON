"""
24. Creating distplot to visualize the distribution of a single variable along with a KDE plot and histogram.

seaborn.distplot: https://seaborn.pydata.org/generated/seaborn.distplot.html
pandas.DataFrame: https://pandas.pydata.org/docs/getting_started/intro_tutorials/

"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Create sample data
data = {'values': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
df = pd.DataFrame(data)

# Create a distribution plot with Seaborn
sns.set_style("whitegrid")
sns.distplot(df['values'], kde=True, hist=True)  # kde for kernel density estimation, hist for histogram

# Add a title and axis labels (optional)
plt.title("Distribution of Values")
plt.xlabel("Value")
plt.ylabel("Density")

# Display the plot
plt.show()
