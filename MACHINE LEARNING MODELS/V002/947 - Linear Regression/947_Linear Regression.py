# File name: 947_Linear Regression.py
# @author Manuela Cortes Granados - 14 Abril 2024 1:42 PM

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Sample data generation
np.random.seed(0)
X = 2 * np.random.rand(100, 1)  # Generate 100 random points between 0 and 2
y = 3 * X + 1 + np.random.randn(100, 1)  # Linear relationship with some random noise

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Plot the results
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_pred, color='red', linewidth=3)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression')
plt.show()
