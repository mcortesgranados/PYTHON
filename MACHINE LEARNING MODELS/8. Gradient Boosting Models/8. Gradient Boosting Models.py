# Importing necessary libraries
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# Load the Boston housing dataset
boston = fetch_openml(data_id=531)
X, y = pd.DataFrame(boston.data), pd.DataFrame(boston.target)

# Rename the columns for better understanding
X.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

# Convert categorical columns to one-hot encoded representation
cat_columns = ['CHAS', 'RAD']
X_encoded = pd.get_dummies(X, columns=cat_columns, drop_first=True)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Defining the XGBoost model
xg_reg = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1,
                           max_depth=5, alpha=10, n_estimators=10)

# Fitting the model to the training data
xg_reg.fit(X_train, y_train)

# Predicting on the test data
preds = xg_reg.predict(X_test)

# Calculating the Mean Squared Error (MSE)
mse = mean_squared_error(y_test, preds)
print("Mean Squared Error:", mse)
