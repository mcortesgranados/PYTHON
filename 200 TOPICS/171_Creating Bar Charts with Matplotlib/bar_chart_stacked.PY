# bar_chart_stacked.py

import matplotlib.pyplot as plt

# Data
categories = ['A', 'B', 'C', 'D', 'E']
values1 = [20, 35, 30, 25, 40]
values2 = [15, 25, 20, 30, 35]

# Create a stacked bar chart
plt.bar(categories, values1, label='Group 1')
plt.bar(categories, values2, bottom=values1, label='Group 2')

# Customize the chart
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Stacked Bar Chart')
plt.legend()

# Display the chart
plt.show()
