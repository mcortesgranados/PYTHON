# line_chart_customized.py

import matplotlib.pyplot as plt
from data import x, y

# Create a line chart with customization
plt.plot(x, y, marker='o', linestyle='--', color='r', label='Data')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Customized Line Chart')
plt.legend()
plt.grid(True)
plt.show()
