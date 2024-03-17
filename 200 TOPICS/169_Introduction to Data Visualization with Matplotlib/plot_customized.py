import matplotlib.pyplot as plt

# Create data
x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]

# Customize plot
plt.plot(x, y, marker='o', linestyle='--', color='r', label='Data')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Customized Plot')
plt.legend()
plt.grid(True)
plt.show()
