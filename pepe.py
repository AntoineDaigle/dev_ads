import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def fit_2d_gaussian_to_data(x, y, z, initial_guess):
    """
    Fit a 2D Gaussian function to a 2D array of data.

    Parameters:
    - x, y: 2D arrays of x and y coordinates.
    - z: 2D array of z values corresponding to x and y coordinates.
    - initial_guess: Initial guess for the parameters of the Gaussian function (amplitude, xo, yo, sigma_x, sigma_y).

    Returns:
    - popt: Optimal values for the parameters of the Gaussian function.
    """

    # Flatten the input arrays
    x_data_flat = x.ravel()
    y_data_flat = y.ravel()
    z_data_flat = z.ravel()

    # Define the 2D Gaussian function
    def gaussian(data, amplitude, xo, yo, sigma_x, sigma_y):
        x, y = data
        g = amplitude * np.exp(-((x - xo) ** 2 / (2 * sigma_x ** 2) + (y - yo) ** 2 / (2 * sigma_y ** 2)))
        return g.ravel()

    # Perform the fit
    popt, pcov = curve_fit(gaussian, (x_data_flat, y_data_flat), z_data_flat, p0=initial_guess)

    return popt

# Generate sample data
x, y = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
amplitude_true = 1.0
xo_true = 0.0
yo_true = 0.0
sigma_x_true = 1.0
sigma_y_true = 0.5
z_true = amplitude_true * np.exp(-((x - xo_true) ** 2 / (2 * sigma_x_true ** 2) + (y - yo_true) ** 2 / (2 * sigma_y_true ** 2)))

# Add noise to the data
z_data = z_true + np.random.normal(scale=0.1, size=z_true.shape)

# Initial guess for parameters
initial_guess = (0.5, 0.0, 0.0, 0.5, 0.5)

# Fit the Gaussian function to the data
optimal_params = fit_2d_gaussian_to_data(x, y, z_data, initial_guess)

# Define the 2D Gaussian function with the optimal parameters
def gaussian(x, y, amplitude, xo, yo, sigma_x, sigma_y):
    g = amplitude * np.exp(-((x - xo) ** 2 / (2 * sigma_x ** 2) + (y - yo) ** 2 / (2 * sigma_y ** 2)))
    return g

# Plotting
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

# Original 2D Gaussian function
ax[0].imshow(z_true, extent=(-5, 5, -5, 5), origin='lower', cmap='viridis')
ax[0].set_title('Original 2D Gaussian')

# Noisy data
ax[1].imshow(z_data, extent=(-5, 5, -5, 5), origin='lower', cmap='viridis')
ax[1].set_title('Noisy Data')

# Fitted Gaussian function
fitted_gaussian = gaussian(x, y, *optimal_params).reshape(x.shape)
ax[2].imshow(fitted_gaussian, extent=(-5, 5, -5, 5), origin='lower', cmap='viridis')
ax[2].set_title('Fitted 2D Gaussian')

plt.show()
