import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.constants import pi
from scipy.interpolate import interp1d
import matplotlib.animation as animation

colors = ['#003D5B', '#D1495B', '#EDAE49', '#00798C', '#401F3E'] 

class DiffusionSimulator():

    def __init__(self, t:int, dt:int, dx:int, D:float):
        """Class that simulate de diffusion of a set of particule.

        Args:
            t (int): Total time of the simulation in seconds.
            dt (int): Time increment in second.
            dx (int): Localisation precision in meters.
            D (float): Diffusion coefficient.
        """
        self.TotalTime = t
        self.TimeSteps = dt
        self.LocalizationPrecision = dx
        self.DiffusionCoefficient = D


    def generation_motion_rectangle(self, x_limit:float=1e-6,
                                    y_limit:float=1e-6):

        x_space = np.linspace(0, 1e-6, 1000)
        n = int(self.TotalTime/self.TimeSteps)

        concentration = 1/np.sqrt(4*pi*self.DiffusionCoefficient*self.TimeSteps)*np.exp(-x_space**2/(4*self.DiffusionCoefficient*self.TimeSteps))

        concentration/=np.sum(concentration)

        x, y = np.zeros(n), np.zeros(n)
        old_x, old_y = 0, 0


        for i in range(n):
            angle = np.random.uniform(0,2*pi)
            r = np.random.choice(a=x_space, p=concentration)
            new_x = old_x + r*np.cos(angle)
            new_y = old_y + r*np.sin(angle)

            if new_x > x_limit/2:
                temp_y = interp1d([old_x, new_x], [old_y, new_y])(x_limit/2)
                temp_x = x_limit/2
                initial_to_wall = np.sqrt((temp_y - old_y)**2 + (temp_x - old_x)**2)

                final_dist_remaining = r - initial_to_wall
                new_angle = np.pi - angle

                new_x = temp_x + final_dist_remaining*np.cos(new_angle)
                new_y = temp_y + final_dist_remaining*np.sin(new_angle)

            if new_x < -x_limit/2:

                temp_y = interp1d([old_x, new_x], [old_y, new_y])(-x_limit/2)
                temp_x = -x_limit/2
                initial_to_wall = np.sqrt((temp_y - old_y)**2 + (temp_x - old_x)**2)

                final_dist_remaining = r - initial_to_wall
                new_angle = (np.pi) - angle

                new_x = temp_x + final_dist_remaining*np.cos(new_angle)
                new_y = temp_y + final_dist_remaining*np.sin(new_angle)

            x[i], old_x = new_x, new_x
            y[i], old_y = new_y, new_y

        self.data = (x,y)

    def MeanSquareDisplacement(self):
        """Method that calculate the mean square displacement.

        Returns:
            np.array: Array containing the MSD
        """
        X, Y= self.data
        MSD = []
        for i in range(2,X.shape[0]):
            x = X[:i]
            y = Y[:i]
            r = np.sqrt(x**2 + y**2)
            msd = np.diff(r)**2
            MSD.append(np.mean(msd))
        return np.cumsum(np.array(MSD))
    

Simulation = DiffusionSimulator(t=40, dt=0.01, dx=1, D=1e-9)
time_array = np.linspace(0,Simulation.TotalTime,
                         int(Simulation.TotalTime/Simulation.TimeSteps))
# Simulation.GenerateMotion()   # Uncomment for og
Simulation.generation_motion_rectangle(1e-6, 1e-6) # uncomment this for limits
x,y = Simulation.data
plt.plot(x,y,color = colors[1])
plt.title("Free 2D diffusion")
plt.xlabel("Position [m]")
plt.ylabel("Position [m]")
plt.axvline(1e-6/2)
plt.axvline(-1e-6/2)
plt.gca().ticklabel_format(style='sci', scilimits=(0, 0))
plt.show()


def calculate_2d_msd(x_positions, y_positions):
    """
    Calculate the Mean Square Displacement (MSD) from 2D arrays of positions.

    Parameters:
    - x_positions: 1D array of x positions over time.
    - y_positions: 1D array of y positions over time.

    Returns:
    - msd_values: 1D array of MSD values over time.
    """
    n = len(x_positions)
    msd_values = np.empty(n - 1)

    for s in range(1, n):
        dx = x_positions[:s] - x_positions[0]
        dy = y_positions[:s] - y_positions[0]
        msd_values[s - 1] = np.mean(dx**2 + dy**2)

    return msd_values


def calc_msd_np(x):
    n = len(x)
    msd = np.empty(n // 4 - 1)
    for s in range(1, n // 4):
        dx = x[s:] - x[:-s]
        msd[s - 1] = np.mean(dx**2)
    return msd


def msd_1d(x):
    result = np.zeros_like(x)
    for i in range(1,len(x)):
        result[i] = np.average((x[i:] - x[:-i])**2)
    return result


def MeanSquaredDisplacement(position):
    squared_position = np.square(position)
    msd = np.empty(len(position))
    for i in range(1, 1 + len(position)):
        msd[i-1] = np.mean(squared_position[:i])
    return msd

fig, (ax1, ax2) = plt.subplots(2, sharex=True)
ax1.plot(time_array, MeanSquaredDisplacement(y))
ax2.plot(time_array, MeanSquaredDisplacement(x))
plt.show()

# def MeanSquaredDisplacement2D(position_x, position_y):
#     squared_position_x = np.square(position_x)
#     squared_position_y = np.square(position_y)
#     r = np.sqrt(squared_position_x + squared_position_y)
#     msd = np.empty(len(position_x))

#     for i in range(1, 1 + len(position_x)):
#         msd[i-1] = np.mean(r[:i])
#     return msd

# plt.plot(time_array, MeanSquaredDisplacement2D(x, y))
# plt.show()
# MSD = Simulation.MeanSquareDisplacement()

# def droite(x, a, b):
#     return a*x*4 + b
# popt, pcov = curve_fit(droite, range(len(pepe)), pepe)

# plt.plot(range(len(pepe)), pepe)
# plt.plot(range(len(pepe)), droite(range(len(pepe)), *popt))
# plt.xlabel("Time [s]")
# plt.ylabel(r"MSD [$\mu m^2$]")
# plt.minorticks_on()
# plt.show()

# print(f"Le coefficient de diffusion est de ({popt[0]} ± {np.diag(pcov)[0]}).")
