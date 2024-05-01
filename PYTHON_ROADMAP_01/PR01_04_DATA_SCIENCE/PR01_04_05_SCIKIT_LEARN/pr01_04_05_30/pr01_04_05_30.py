"""
30. Handling Missing Data: Dealing with missing values in datasets.

Handling missing data is a common task in data preprocessing, as many real-world datasets contain missing values. These missing 
values can adversely affect the performance of machine learning algorithms if not handled properly. In scikit-learn, 
missing data can be handled using the SimpleImputer class from the sklearn.impute module. This class provides various strategies 
for imputing missing values, such as replacing them with the mean, median, most frequent value, or a constant.

Let's see an example of how to handle missing data using the SimpleImputer class:

Explanation:

We import the necessary libraries, including numpy, pandas, and SimpleImputer from sklearn.impute.

We create a sample dataset with missing values. In this example, we use a dictionary to create a pandas DataFrame.

We display the original DataFrame to observe the missing values.

We initialize the SimpleImputer with the strategy set to 'mean'. This means that missing values will be replaced with the mean of each column.

We use the fit_transform method of the SimpleImputer object to impute missing values in the DataFrame.

The imputed data is returned as a NumPy array. We convert this array back to a pandas DataFrame with the same column names.

We display the DataFrame after handling missing values to observe the changes.

In this example, missing values in each column are replaced with the mean of that column. However, you can replace missing values with other strategies 
such as median, most frequent value, or a constant by specifying the strategy parameter accordingly when initializing the SimpleImputer. 
Handling missing data is essential for building accurate and reliable machine learning models, and scikit-learn provides a convenient way to perform this task.


"""

# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

# Sample dataset with missing values
data = {
    'A': [1, 2, np.nan, 4, 5],
    'B': [6, np.nan, 8, 9, 10],
    'C': [np.nan, 12, 13, np.nan, 15]
}

# Convert the dictionary to a pandas DataFrame
df = pd.DataFrame(data)

# Display the original DataFrame
print("Original DataFrame:")
print(df)

# Initialize the SimpleImputer with strategy='mean'
imputer = SimpleImputer(strategy='mean')

# Impute missing values
imputed_data = imputer.fit_transform(df)

# Convert the imputed data back to a DataFrame
imputed_df = pd.DataFrame(imputed_data, columns=df.columns)

# Display the DataFrame after imputation
print("\nDataFrame after handling missing values:")
print(imputed_df)
