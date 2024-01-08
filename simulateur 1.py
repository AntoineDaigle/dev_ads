import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import matplotlib.animation as animation
from scipy.constants import pi

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


    def generation_motion_circle(self, radius = np.inf):
        n = int(self.TotalTime/self.TimeSteps)
        x, y = np.zeros(n), np.zeros(n)
        old_x, old_y = 0,0
        factor = np.sqrt(2*self.DiffusionCoefficient*self.TimeSteps)

        for i in range(n):
            r1 =  np.random.normal(0,factor) + np.random.normal(0,self.LocalizationPrecision)
            r2 =  np.random.normal(0,factor) + np.random.normal(0,self.LocalizationPrecision)
            new_x = old_x + r1
            new_y = old_y + r2

            if np.abs(new_x**2+new_y**2) > radius**2:
                r = np.sqrt(r1**2 + r2**2)
                angle = np.arctan(r2/r1)

                temp_y = np.linspace(old_y,new_y)
                temp_x =np.linspace(old_x,new_x)

                radiuses = np.sqrt(temp_y**2+temp_x**2)

                for j in range(len(radiuses)):
                    rad = radiuses[j]
                    if rad>radius:
                        break
                temp_x,temp_y = temp_x[j], temp_y[j]
                initial_to_wall = np.sqrt((temp_y - old_y)**2 + (temp_x - old_x)**2)
                final_dist_remaining = r - initial_to_wall
                #conditions selon le cadran
                phi = np.arctan(temp_y/temp_x)
                if temp_y>0 and temp_x>0:
                    new_angle = 2*phi - angle + np.pi
                if temp_y>0 and temp_x<0:
                    new_angle = 2*phi - angle
                if temp_y<0 and temp_x<0:
                    new_angle = 2*phi - angle
                if temp_y<0 and temp_x>0:
                    new_angle = 2*phi - angle + np.pi
                new_x = temp_x + final_dist_remaining*np.cos(new_angle)
                new_y = temp_y + final_dist_remaining*np.sin(new_angle)

                if np.abs(new_x**2+new_y**2) > radius**2:
                    new_x = temp_x
                    new_y = temp_y
            x[i], old_x = new_x, new_x
            y[i], old_y = new_y, new_y
        self.data = (x,y)


    def generation_motion_rectangle(self, x_limit:float=1e-6):

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
                temp_y = interp1d([old_x, new_x], [old_y, new_y],fill_value="extrapolate")(x_limit/2)
                temp_x = x_limit/2
                initial_to_wall = np.sqrt((temp_y - old_y)**2 + (temp_x - old_x)**2)

                final_dist_remaining = r - initial_to_wall
                new_angle = np.pi - angle

                new_x = temp_x + final_dist_remaining*np.cos(new_angle)
                new_y = temp_y + final_dist_remaining*np.sin(new_angle)


            if new_x < -x_limit/2:
                temp_y = interp1d([old_x, new_x], [old_y, new_y],fill_value="extrapolate")(-x_limit/2)
                temp_x = -x_limit/2
                initial_to_wall = np.sqrt((temp_y - old_y)**2 + (temp_x - old_x)**2)
                final_dist_remaining = r - initial_to_wall
                new_angle = 2 * np.pi - angle

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
        MSD = np.empty(X.shape[0])
        for i in range(1,X.shape[0]):
            xsquared = np.diff(X[:i])**2
            ysquared = np.diff(Y[:i])**2
            msd = np.mean(np.sum(xsquared + ysquared))
            MSD[i] = msd
        return MSD[1:]




if __name__ == "__main__":


    ################
    ### Numéro 4 ###
    ################
    dt=0.01
    D=2.5e-13
    t=20
    dx=2e-9
    Simulation = DiffusionSimulator(t=t, dt=dt, dx=dx, D=D)
    time_array = np.linspace(0,Simulation.TotalTime,
                            int(Simulation.TotalTime/Simulation.TimeSteps))
    Simulation.generation_motion_rectangle(1e-6)
    x,y = Simulation.data


    x_msd = np.empty(x.shape[0])
    y_msd = np.empty(x.shape[0])
    for i in range(1,x_msd.shape[0]):
        xsquared = np.diff(x[:i])**2
        ysquared = np.diff(y[:i])**2
        xsquared = x[:i]**2
        ysquared = y[:i]**2
        x_msd[i] = np.mean(np.sum(xsquared))
        y_msd[i] = np.mean(np.sum(ysquared))
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(x, y)
    ax1.set_xlabel("Position [m]")
    ax1.set_ylabel("Position [m]")
    # ax1.add_patch(plt.Circle((0,0),2e-6,fill=False,color = colors[4]))
    ax1.minorticks_on()

    ax2.plot(time_array, x_msd, label="x component")
    ax2.plot(time_array, y_msd, label="y component")
    ax2.legend()
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("MSD")
    ax2.minorticks_on()
    plt.tight_layout()
    # plt.savefig("figures/simulateur1/numéro4.pdf")
    plt.show()
