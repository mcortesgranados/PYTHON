"""
06. Generating pie charts to represent the composition of a categorical variable.

"""

import matplotlib.pyplot as plt

# Sample data
categories = ['A', 'B', 'C', 'D']
sizes = [25, 35, 20, 20]
colors = ['lightblue', 'lightgreen', 'lightsalmon', 'lightpink']

# Create a pie chart
plt.pie(sizes, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)

# Equal aspect ratio ensures that pie is drawn as a circle
plt.axis('equal')

# Add title
plt.title('Pie Chart Example')

# Show plot
plt.show()
