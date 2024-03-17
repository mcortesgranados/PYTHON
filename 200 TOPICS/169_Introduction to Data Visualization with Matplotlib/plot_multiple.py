import matplotlib.pyplot as plt

# Create data
x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]

# Create subplots
fig, axs = plt.subplots(2, 1, figsize=(8, 6))

# Plot data on each subplot
axs[0].plot(x, y)
axs[0].set_title('Plot 1')

axs[1].scatter(x, y, color='g', marker='o')
axs[1].set_title('Plot 2')

# Adjust layout and display
plt.tight_layout()
plt.show()
