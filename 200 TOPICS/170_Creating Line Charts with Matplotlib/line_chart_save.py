# line_chart_save.py

import matplotlib.pyplot as plt
from data import x, y

# Create a line chart
plt.plot(x, y)

# Customize the chart
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Line Chart')

# Save the chart as an image file
plt.savefig('line_chart.png')
