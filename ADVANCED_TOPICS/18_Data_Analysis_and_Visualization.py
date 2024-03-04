# FileName: 18_Data_Analysis_and_Visualization.py
# Code Written by Manuel Cortés Granados
# Date: 2024-03-04
# Location: Bogota DC, Colombia
# LinkedIn Profile: https://www.linkedin.com/in/mcortesgranados/

# This Python code demonstrates data analysis and visualization using pandas, matplotlib, and seaborn libraries.
# It loads a sample dataset (Iris dataset), performs data analysis by displaying summary statistics,
# and visualizes the data using pairplot and boxplot.
# Additionally, it includes metadata such as the proposed filename, authorship information, date and time, location,
# and a link to the author's LinkedIn profile for context.

# Installation of dependencies:
# You may need to install pandas, matplotlib, and seaborn libraries.
# Use the following commands to install them:
# pip install pandas
# pip install matplotlib
# pip install seaborn
# pip install pygments  

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load sample dataset
iris = sns.load_dataset('iris')

# Display the first few rows of the dataset
print("First few rows of the Iris dataset:")
print(iris.head())

# Data Analysis
print("\nData Analysis:")
print("Summary statistics:")
print(iris.describe())

# Data Visualization
print("\nData Visualization:")
# Pairplot for pairwise relationships between features
sns.pairplot(iris, hue='species', palette='Set2')
plt.title('Pairplot of the Iris dataset')
plt.show()

# Boxplot for distribution of features by species
plt.figure(figsize=(10, 6))
sns.boxplot(data=iris, x='species', y='sepal_length')
plt.title('Boxplot of Sepal Length by Species')
plt.show()
