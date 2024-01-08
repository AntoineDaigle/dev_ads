import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.constants import pi
from scipy.interpolate import interp1d
import matplotlib.animation as animation

colors = ['#003D5B', '#D1495B', '#EDAE49', '#00798C', '#401F3E'] 

class DiffusionSimulator():

    def __init__(self, t:float, dt:float, dx:float, D:float):
        """Class that simulate de diffusion of a set of particule.

        Args:
            t (float): Total time of the simulation in seconds.
            dt (float): Time increment in second.
            dx (float): Localisation precision in meters.
            D (float): Diffusion coefficient.
        """
        self.TotalTime = t
        self.TimeSteps = dt
        self.LocalizationPrecision = dx
        self.DiffusionCoefficient = D

    def generation_motion_rectangle(self, x_limit:float=1e-6,
                                    y_limit:float=1e-6):
        
        n = int(self.TotalTime/self.TimeSteps)
        x, y = np.zeros(n), np.zeros(n)
        old_x, old_y = 0, 0
        factor = np.sqrt(2*self.DiffusionCoefficient*self.TimeSteps)
        for i in range(n):
            r1 =  np.random.normal(0,factor) + np.random.normal(0,self.LocalizationPrecision)
            r2 =  np.random.normal(0,factor) + np.random.normal(0,self.LocalizationPrecision)
            new_x = old_x + r1
            new_y = old_y + r2
            r = np.sqrt(r1**2 + r2**2)
            angle = np.arctan(r2/r1)

            if new_x > x_limit/2:
                angle_normal = np.arctan(new_y/new_x)
                temp_y = interp1d([old_x, new_x], [old_y, new_y],fill_value="extrapolate")(x_limit/2)
                temp_x = x_limit/2
                initial_to_wall = np.sqrt((temp_y - old_y)**2 + (temp_x - old_x)**2)

                final_dist_remaining = r - initial_to_wall
                new_angle = angle_normal - angle

                new_x = temp_x + final_dist_remaining*np.cos(new_angle)
                new_y = temp_y + final_dist_remaining*np.sin(new_angle)

            if new_x < -x_limit/2:
                angle_normal = np.arctan(new_y/new_x)
                temp_y = interp1d([old_x, new_x], [old_y, new_y],fill_value="extrapolate")(-x_limit/2)
                temp_x = -x_limit/2
                initial_to_wall = np.sqrt((temp_y - old_y)**2 + (temp_x - old_x)**2)

                final_dist_remaining = r - initial_to_wall
                new_angle =  angle_normal - angle

                new_x = temp_x + final_dist_remaining*np.cos(new_angle)
                new_y = temp_y + final_dist_remaining*np.sin(new_angle)
            
            new_x = np.clip(new_x,-x_limit/2,x_limit/2)
            new_y = np.clip(new_y,-y_limit/2,y_limit/2)
            x[i], old_x = new_x, new_x
            y[i], old_y = new_y, new_y

        self.data = (x,y)

    def MeanSquareDisplacement(self):
        """Method that calculate the mean square displacement.

        Returns:
            np.array: Array containing the MSD
        """
        X, Y= self.data
        MSD = np.empty(X.shape[0])
        for i in range(1,X.shape[0]):
            xsquared = np.diff(X[:i])**2
            ysquared = np.diff(Y[:i])**2
            msd = np.mean(np.sum(xsquared + ysquared))
            MSD[i] = msd
        return MSD[1:]
    


def MSD1D(X,Y):
    """
    Calculate the Mean Square Displacement (MSD) from 2D arrays of positions.

    Parameters:
    - x_positions: 1D array of x positions over time.
    - y_positions: 1D array of y positions over time.

    Returns:
    - msd_values: 1D array of MSD values over time.
    """
    MSDX,MSDY = np.empty(X.shape[0]),np.empty(Y.shape[0])
    for i in range(1,X.shape[0]):
        xsquared = np.diff(X[:i])**2
        ysquared = np.diff(Y[:i])**2
        msd_x = np.mean(np.sum(xsquared))
        msd_y = np.mean(np.sum(ysquared))
        MSDX[i] = msd_x
        MSDY[i] = msd_y
    return MSDX, MSDY


def droite(x,a,b):
    return 4*a*x + b
Diff = []

# T = np.linspace(2,30)
T = np.linspace(1,200)
T = [50]
for t in T:
    Ds = []
    for _ in range(1):
        dt=0.01
        D=2.5e-13
        # t=20
        dx=2e-9
        Simulation = DiffusionSimulator(t=t, dt=dt, dx=dx, D=D)
        time_array = np.linspace(0,Simulation.TotalTime,
                                int(Simulation.TotalTime/Simulation.TimeSteps))
        # Simulation.GenerateMotion()   # Uncomment for og
        Simulation.generation_motion_rectangle(1e-6,np.inf) # uncomment this for limits
        x,y = Simulation.data
        plt.plot(x*1e6,y*1e6,color = colors[0])
        plt.title("Free 2D diffusion")
        plt.xlabel(r"Position [$\mu$m]")
        plt.ylabel(r"Position [$\mu$m]")
        # plt.axvline(1/2)
        # plt.axvline(-1/2)
        plt.gca().ticklabel_format(style='sci', scilimits=(0, 0))
        # plt.savefig('Diffusion_example_rec.pdf')
        plt.show()

        msd = Simulation.MeanSquareDisplacement()
        msd_2x, msd_2y = MSD1D(x,y)
        # msd_2 = np.cumsum(msd_2)

        fit,pcov = curve_fit(droite,time_array[1:],msd)
        d = fit[0]
        Ds.append(d)
    Diff.append(np.mean(Ds))
        # fit2,pcov = curve_fit(droite,time_array[1:],msd_2)
        # print(fit2/(4*dt))


    plt.plot(time_array[1:], msd*(1e6)**2,color = colors[0], label = 'Simulation')
    plt.plot(time_array,msd_2x*(1e6)**2)
    plt.plot(time_array,msd_2y*(1e6)**2)
    plt.plot(time_array, droite(time_array*(1e6)**2,d,fit[1]), color = colors[2], ls = '--',alpha=0.6, label = f'MSD=4Dt')
    plt.grid(alpha = 0.9, ls='--')
    plt.xlabel('Temps (s)')
    plt.legend()
    plt.ylabel(r'MSD ($\mu$m$^2$)')
    # plt.savefig('curve_fit_rec.pdf')
    # plt.plot(time_array[1:],msd_2)
    plt.show()
print(d)

# plt.scatter(T,Diff,color = colors[0])
# plt.xlabel('Temps de simulation (s)')
# plt.ylabel(r'Estimation de D (m$^2$/s)')
# plt.savefig('1B.pdf')
# plt.show()

def MeanSquaredDisplacement2D(position_x, position_y):
    squared_position_x = np.square(position_x)
    squared_position_y = np.square(position_y)
    r = np.sqrt(squared_position_x + squared_position_y)
    msd = np.empty(len(position_x))

    for i in range(1, 1 + len(position_x)):
        msd[i-1] = np.mean(r[:i])
    return msd

# Simulation = DiffusionSimulator(t=40, dt=0.01, dx=1, D=1e-9)
# time_array = np.linspace(0,Simulation.TotalTime,
#                         int(Simulation.TotalTime/Simulation.TimeSteps))
# # Simulation.GenerateMotion()   # Uncomment for og
# Simulation.generation_motion_rectangle(np.inf,np.inf) # uncomment this for limits
# x,y = Simulation.data
# MSD = MeanSquaredDisplacement2D(x,y)

# def droite(x, a, b):
#     return a*x*4 + b

# popt, pcov = curve_fit(droite, time_array, MSD)

# plt.plot(time_array, MSD)
# # plt.plot(range(len(pepe)), droite(range(len(pepe)), *popt))
# plt.xlabel("Time [s]")
# plt.ylabel(r"MSD [$\mu m^2$]")
# plt.minorticks_on()
# plt.show()

# print(f"Le coefficient de diffusion est de ({popt[0]} ± {np.diag(pcov)[0]}).")