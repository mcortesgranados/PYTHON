# bar_chart.py

import matplotlib.pyplot as plt

# Data
categories = ['A', 'B', 'C', 'D', 'E']
values = [20, 35, 30, 25, 40]

# Create a bar chart
plt.bar(categories, values)

# Customize the chart
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Bar Chart')

# Display the chart
plt.show()
