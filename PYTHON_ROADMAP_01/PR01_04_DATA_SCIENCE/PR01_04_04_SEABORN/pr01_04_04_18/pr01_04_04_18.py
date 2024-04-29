"""
18. Creating point plots to compare values of one variable across different levels of another variable.

"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Create sample data
data = {'treatment': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
        'value': [2, 1, 4, 5, 3, 7, 1, 2, 4],
        'group': ['control', 'control', 'control', 'treated', 'treated', 'treated', 'treated', 'treated', 'treated']}
df = pd.DataFrame(data)

# Create a point plot using Seaborn
sns.set_style("whitegrid")
ax = sns.pointplot(x="treatment", y="value", hue="group", data=df)
ax.set_title("Comparison of Values by Treatment Group")
ax.set_xlabel("Treatment")
ax.set_ylabel("Value")

# Customize the plot (optional)
plt.legend(title="Group")  # Add legend title

# Display the plot
plt.show()
