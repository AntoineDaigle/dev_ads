import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

# Generate some sample 2D data
np.random.seed(0)
mean = [0, 0]
cov = [[1, 0.5], [0.5, 1]]
data = np.random.multivariate_normal(mean, cov, 1000)

# Fit a Gaussian to the data
mu, cov = multivariate_normal.fit(data)

# Generate new points based on the fitted Gaussian
x, y = np.mgrid[-3:3:.01, -3:3:.01]
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x
pos[:, :, 1] = y
rv = multivariate_normal(mu, cov)
z = rv.pdf(pos)

# Plot the original data and the fitted Gaussian
plt.figure(figsize=(10, 5))
plt.scatter(data[:, 0], data[:, 1], alpha=0.5, label='Original Data')
plt.contour(x, y, z, levels=10, cmap='viridis', label='Fitted Gaussian')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('2D Gaussian Fit')
plt.legend()
plt.colorbar()
plt.show()
