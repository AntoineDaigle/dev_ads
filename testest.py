import numpy as np
import matplotlib.pyplot as plt


def g_function(data, diff_coeff, N):
    tau = 8e-4
    wo = 2*25e-9
    g = (0.3535 / N) * 1/((1 + 4 * diff_coeff * tau * data / np.square(wo))**3/2)
    return g

x_range = np.linspace(0, 10)

plt.plot(x_range, g_function(x_range, 1e-12, 1))
plt.show()