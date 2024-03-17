import matplotlib.pyplot as plt

# Create data
x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]

# Bar plot
plt.bar(x, y)
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Bar Plot')
plt.show()

# Histogram
plt.hist(y, bins=3)
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.show()

# Scatter plot
plt.scatter(x, y)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Scatter Plot')
plt.show()
