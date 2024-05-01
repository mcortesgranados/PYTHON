import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load sample data (replace with your actual dataset)
data = pd.read_csv("your_data.csv")  # Assuming CSV format, adjust for other formats

# Define features (independent variables) and target variable (dependent variable)
features = ["feature1", "feature2", ...]  # Replace with actual feature names in your data
target = "target_variable"

# Split data into training and testing sets (typically 80/20 split)
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate model performance using mean squared error (MSE)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# (Optional) Print model coefficients for interpretation
# Coefficients provide insights into the relationship between features and target
print("Model coefficients:", model.coef_)
print("Model intercept:", model.intercept_)
