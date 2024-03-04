# FileName: 27_Data_Analysis_and_Visualization.py
# Code Written by Manuel Cortés Granados
# Date: 2024-03-04
# Location: Bogota DC, Colombia
# LinkedIn Profile: https://www.linkedin.com/in/mcortesgranados/

# Data Analysis and Visualization with Python

# Python offers powerful libraries such as pandas for data manipulation and analysis, and matplotlib and seaborn for data visualization.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Example: Analyzing and visualizing a dataset using pandas, matplotlib, and seaborn

# Load the Iris dataset using seaborn's load_dataset function
# The 'iris' dataset is a classic dataset in machine learning and statistics,
# containing measurements of iris flowers and their corresponding species (setosa, versicolor, or virginica)
# It is commonly used for learning and demonstrating various data analysis and machine learning techniques
df = sns.load_dataset('iris')

# Display basic information about the dataset
print("Basic information about the dataset:")
print(df.info())

# Display summary statistics
print("\nSummary statistics of numerical columns:")
print(df.describe())

# Visualize data using seaborn pairplot
sns.pairplot(df, hue='species', height=2.5)
plt.title("Pairplot of Iris Dataset")
plt.show()
