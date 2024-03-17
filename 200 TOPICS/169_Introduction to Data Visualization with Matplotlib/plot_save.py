import matplotlib.pyplot as plt

# Create data
x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]

# Plot data
plt.plot(x, y)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Saved Plot')
plt.savefig('plot.png')
