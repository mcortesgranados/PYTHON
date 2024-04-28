"""
29. Generating images with imshow to display 2D data as an image.

"""

import numpy as np
import matplotlib.pyplot as plt

# Generate random 2D array (image data)
image_data = np.random.rand(10, 10)

# Plot the image using imshow
plt.imshow(image_data, cmap='viridis', interpolation='nearest')
plt.colorbar()  # Add colorbar to indicate intensity scale
plt.title('2D Image Visualization')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
