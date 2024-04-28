"""
01. Creating line plots to visualize trends or relationships between variables.
python .\PYTHON_ROADMAP_01\PR01_04_DATA_SCIENCE\PR01_04_03_MATPLOTLIB\pr01_04_03_01\pr01_04_03_01.py

"""
import matplotlib.pyplot as plt

# Sample data
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# Create a line plot
plt.plot(x, y, marker='o', linestyle='-', color='b', label='Line Plot')

# Add labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Line Plot Example')

# Add grid
plt.grid(True)

# Add legend
plt.legend()

# Show plot
plt.show()
