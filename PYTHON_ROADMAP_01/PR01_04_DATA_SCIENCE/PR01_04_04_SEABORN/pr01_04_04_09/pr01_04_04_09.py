"""
09. Generating pair plots to explore pairwise relationships between variables in a dataset.

"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Sample dataset
data = {
    'A': [1, 2, 3, 4, 5],
    'B': [2, 4, 6, 8, 10],
    'C': [3, 6, 9, 12, 15]
}
df = pd.DataFrame(data)

# Create pair plot
sns.pairplot(df)

# Show plot
plt.show()
