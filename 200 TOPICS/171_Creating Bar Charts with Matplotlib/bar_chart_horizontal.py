# bar_chart_horizontal.py

import matplotlib.pyplot as plt

# Data
categories = ['A', 'B', 'C', 'D', 'E']
values = [20, 35, 30, 25, 40]

# Create a horizontal bar chart
plt.barh(categories, values)

# Customize the chart
plt.xlabel('Values')
plt.ylabel('Categories')
plt.title('Horizontal Bar Chart')

# Display the chart
plt.show()
