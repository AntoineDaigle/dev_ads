import numpy as np
import matplotlib.pyplot as plt

# Create a sample 2D NumPy array (replace this with your own image data)
data = np.random.random((10, 10))

# Create a pcolormesh plot
plt.pcolormesh(data, cmap='viridis')

# Add a circle to the plot
circle_radius = 2
circle_center = (5, 5)

circle = plt.Circle(circle_center, circle_radius, color='red', fill=False, linewidth=2)
plt.gca().add_patch(circle)

# Show the plot
plt.colorbar()
plt.show()
