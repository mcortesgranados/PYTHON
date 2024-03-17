# bar_chart_customized.py

import matplotlib.pyplot as plt

# Data
categories = ['A', 'B', 'C', 'D', 'E']
values = [20, 35, 30, 25, 40]

# Create a bar chart with customization
plt.bar(categories, values, color='skyblue', edgecolor='black', linewidth=2, alpha=0.7)

# Customize the chart
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Customized Bar Chart')

# Display the chart
plt.show()
