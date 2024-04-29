"""
21. Creating lmplot to visualize the relationship between two continuous variables with options for grouping by additional categorical variables.

seaborn.residplot: https://seaborn.pydata.org/generated/seaborn.residplot.html (This documentation might not specify supported data types for x and y)
pandas.DataFrame: https://pandas.pydata.org/docs/getting_started/intro_tutorials/
sklearn.linear_model.LinearRegression: https://scikit-

"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Create sample data
data = {'x': [1, 2, 3, 4, 5],
        'y': [3, 5, 7, 2, 8]}
df = pd.DataFrame(data)

# Fit a linear regression model
model = LinearRegression()
model.fit(df[['x']], df['y'])

# Predict y values based on the fitted model
y_pred = model.predict(df[['x']])

# Calculate residuals (actual y - predicted y)
residuals = df['y'] - y_pred

# Generate residual plot using Seaborn
sns.set_style("whitegrid")
ax = sns.residplot(x="x", y="y", data=df)  # x and y can be actual values or predicted vs actual
ax.set_title("Residual Plot - Visualizing Model Errors")
ax.set_xlabel("X-axis Variable")
ax.set_ylabel("Residuals (y - y_predicted)")
plt.show()
