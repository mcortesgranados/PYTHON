"""
17. Generating bar plots with confidence intervals to compare categorical data.

"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Create some sample data
data = {'category': ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
        'value': [2, 1, 4, 5, 3, 7, 1, 2, 4],
        'error': [0.2, 0.3, 0.1, 0.5, 0.2, 0.8, 0.3, 0.1, 0.2]}
df = pd.DataFrame(data)

# Define a function to calculate the confidence interval (CI)
def calculate_ci(data, confidence_level=0.95):
  """
  This function calculates the confidence interval for a given data set.

  Args:
      data (list): A list of data points.
      confidence_level (float, optional): The confidence level for the CI. Defaults to 0.95.

  Returns:
      tuple: A tuple containing the lower and upper bounds of the CI.
  """
  mean = np.mean(data)
  std = np.std(data)
  n = len(data)
  z = np.sqrt(n) * stats.norm.ppf((1 + confidence_level) / 2)
  ci_lower = mean - z * std
  ci_upper = mean + z * std
  return ci_lower, ci_upper

# Create a bar chart with error bars using Seaborn
sns.set_style("whitegrid")
ax = sns.barplot(x="category", y="value", yerr="error", data=df)
ax.set_title("Comparison of Categorical Data with Confidence Intervals")
ax.set_xlabel("Category")
ax.set_ylabel("Value")

# Add confidence intervals (CI) as error bars to the plot manually
categories = df["category"].unique()
for i, cat in enumerate(categories):
  data_points = df[df["category"] == cat]["value"]
  ci_lower, ci_upper = calculate_ci(data_points)
  plt.errorbar(i, df[df["category"] == cat]["value"].mean(), 
              yerr=(ci_upper - ci_lower)/2, fmt='none', ecolor='black', capsize=7)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
