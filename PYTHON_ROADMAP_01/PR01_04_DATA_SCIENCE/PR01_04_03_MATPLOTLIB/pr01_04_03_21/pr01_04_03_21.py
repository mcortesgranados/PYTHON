"""
21. Creating radar charts to display multivariate data in a circular layout.

"""
import numpy as np
import matplotlib.pyplot as plt

# Function to create radar chart
def radar_chart(ax, labels, values, title=None):

    # Number of variables
    num_vars = len(labels)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # Make the plot close to a circle
    values += values[:1]

    # Plot data
    ax.plot(angles, values, linewidth=1, linestyle='solid')

    # Fill area
    ax.fill(angles, values, 'b', alpha=0.1)

    # Set labels and title
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title(title)

# Sample data
labels = ['A', 'B', 'C', 'D', 'E']
values = [4, 3, 2, 5, 4]

# Create figure and axis
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# Create radar chart
radar_chart(ax, labels, values, title='Radar Chart Example')

# Show plot
plt.show()
