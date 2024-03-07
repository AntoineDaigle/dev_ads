import numpy as np
import matplotlib.pyplot as plt

x,y = np.meshgrid(10, 10)

def gs(xy, N, wo, wz, tau_p, tau_l, delta_x, delta_y, D):
    """The component of the ACF due to diffusion is the traditional
    correlation function.

    Args:
        epsilon (int): Spatial lag in pixel on the x axis.
        phi (_type_): Spatiale lag in pizel on the y axis.
        N (_type_): Number of particules.
        wo (_type_): Point spread function of the laser.
        wz (_type_): z-axis beam radius.
        tau_p (_type_): Pixel dwell time in x.
        tau_l (_type_): Pixel dwell time in y.
        D (float): Diffusion coefficient

    Returns:
        float: Traditional correlation function
    """
    epsilon, phi = xy
    g = (0.3535 / N) * 1/(1 + (4 * D * (tau_p * np.abs(epsilon) + tau_l * np.abs(phi)))/wo**2) * 1/np.sqrt(1 + (4 * D * (tau_p * np.abs(epsilon) + tau_l * np.abs(phi)))/wz**2)

    s = np.exp(-1 * ((np.abs(epsilon) * delta_x / wo)**2 + (np.abs(phi) * delta_y /wo)**2)/(1 + (4 * D * (tau_p * np.abs(epsilon) + tau_l *phi)/(wo**2))))
    return (g * s).ravel()



dt = 10e-6 # pixel dwell time
D = 1e-12 # coefficient de diffusion
nx = 20
ny = 20
pixels = (nx,ny) # taille de l'image
psf = 50e-9 # tache du fluorophore / taille du faisceau laser
ps = 25e-9
time = 2
density = 1#0.6#0.02


plt.imshow(gs(xy=(x, y), N=7, wo=50e-9, wz=150e-9, tau_p=10e-6, tau_l=10e-6,
              delta_x=25e-9, delta_y=25e-9, D=1e-12))
plt.show()

