import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.morphology import disk
from scipy import stats
from scipy.optimize import curve_fit

def AutocorrelationFit(data, epsilon, psi, diff_coef):
    center=10
    pos = np.dstack(data)
    tau_p = 8e-6
    tau_l = 20 * tau_p
    N = 7
    wo = 50e-9
    wz = 3*wo
    delta_x = 25e-9
    delta_y = 25e-9

    rep_num = 4 * diff_coef * (tau_p * np.abs(epsilon) + tau_l * np.abs(psi))
    denum = 1 + rep_num/(wo**2)
    a = (0.3535 / N) * 1/(1 + rep_num/(wo**2)) * 1/np.sqrt(1 + rep_num/(wz**2))
    b = (np.abs(epsilon) * delta_x / wo)**2 / denum
    d = (np.abs(psi) * delta_y / wo)**2 / denum
    gaussienne = stats.multivariate_normal([center, center], [[b,0],[0,d]],
                                           allow_singular=True).pdf(pos)
    gaussienne /= np.max(gaussienne)

    return (a*gaussienne).ravel()

xv, yv = np.meshgrid(np.arange(0,20), np.arange(0,20))

plt.imshow(AutocorrelationFit((xv, yv), 1e-6, 1e-6, 1e-12).reshape(20, 20))
plt.show()