"""
30. Creating animated plots to visualize dynamic data or processes over time.

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Initialize figure and axis
fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], 'r-', animated=True)

# Function to initialize the plot
def init():
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(-1, 1)
    return ln,

# Function to update the plot at each frame
def update(frame):
    xdata.append(frame)
    ydata.append(np.sin(frame))
    ln.set_data(xdata, ydata)
    return ln,

# Create the animation
ani = FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 100),
                    init_func=init, blit=True)

# Show the animated plot
plt.show()
