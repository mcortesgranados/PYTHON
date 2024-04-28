"""
07. Building area plots to display the magnitude of changes over time.

"""

import matplotlib.pyplot as plt

# Sample data
years = [2010, 2011, 2012, 2013, 2014]
var1 = [10, 20, 15, 25, 30]
var2 = [5, 15, 10, 20, 25]
var3 = [15, 25, 20, 30, 35]

# Create area plot
plt.stackplot(years, var1, var2, var3, labels=['Variable 1', 'Variable 2', 'Variable 3'], colors=['lightblue', 'lightgreen', 'lightsalmon'])

# Add labels and title
plt.xlabel('Year')
plt.ylabel('Magnitude')
plt.title('Area Plot Example')

# Add legend
plt.legend()

# Show plot
plt.show()
