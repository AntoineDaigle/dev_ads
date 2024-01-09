import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import matplotlib.animation as animation
from tqdm import tqdm
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
                temp_y = interp1d([old_x, new_x], [old_y, new_y],
                                  fill_value="extrapolate")(x_limit/2)
                temp_x = x_limit/2
                initial_to_wall = np.sqrt((temp_y - old_y)**2 + (temp_x - old_x)**2)

                final_dist_remaining = r - initial_to_wall
                new_angle = np.pi - angle

                new_x = temp_x + final_dist_remaining*np.cos(new_angle)
                new_y = temp_y + final_dist_remaining*np.sin(new_angle)


            if new_x < -x_limit/2:
                temp_y = interp1d([old_x, new_x], [old_y, new_y],
                                  fill_value="extrapolate")(-x_limit/2)
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
    ### Numéro 1 ###
    ################
    dt=0.01
    D=2.5e-13
    t=20
    dx=2e-9
    
    msd_array = np.empty((30, int(t/dt)-1))

    for i in range(30):
        Simulation = DiffusionSimulator(t=t, dt=dt, dx=dx, D=D)
        time_array = np.linspace(0,Simulation.TotalTime,
                                int(Simulation.TotalTime/Simulation.TimeSteps))
        Simulation.generation_motion_rectangle(np.inf)
        x,y = Simulation.data
        msd_array[i] = Simulation.MeanSquareDisplacement()

    msd_mean = np.mean(msd_array, axis=0)
    msd_erro = np.std(msd_array, axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot(x, y, c=colors[0])
    ax1.set_aspect('equal')
    ax1.set_xlabel("Position [m]")
    ax1.set_ylabel("Position [m]")
    ax1.minorticks_on()

    def droite(x,a,b):
        return 4*a*x + b

    popt, pcov = curve_fit(droite, time_array[:-1], msd_mean)

    print(f"Le coefficient de diffusion est de ({popt[0]} ± {np.diag(pcov)[0]}).")

    ax2.fill_between(time_array[:-1], msd_mean+msd_erro, msd_mean-msd_erro,
                     color=colors[1], alpha=0.3)
    ax2.plot(time_array[:-1], msd_mean, c=colors[1])
    # ax2.plot(time_array[:-1], Simulation.MeanSquareDisplacement(),
    #          label="Raw data")
    # ax2.plot(time_array, droite(time_array, *popt), label="Curve fit")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("MSD")
    ax2.minorticks_on()
    # ax2.legend()
    plt.tight_layout()
    plt.savefig("figures/simulateur1/numéro1.pdf")
    plt.show()


    # ################
    # ### Numéro 2 ###
    # ################
    # dt=0.01
    # D=2.5e-13
    # t=10
    # dx=2e-9
    # diff_coef = np.empty((30, int(t/dt)))

    # def droite(x,a,b):
    #     return 4*a*x + b

    # for j in tqdm(range(30)):
    #     Simulation = DiffusionSimulator(t=t, dt=dt, dx=dx, D=D)
    #     time_array = np.linspace(0,Simulation.TotalTime,
    #                             int(Simulation.TotalTime/Simulation.TimeSteps))
    #     Simulation.generation_motion_rectangle(np.inf)
    #     x,y = Simulation.data

    #     temp_diff_coeff = np.empty(len(x))

    #     for i in range(3, len(x)):
    #         popt, pcov = curve_fit(droite, time_array[:i],
    #                             Simulation.MeanSquareDisplacement()[:i])
    #         temp_diff_coeff[i] = popt[0]
    #     diff_coef[j, :] = temp_diff_coeff

    # diff_coef_mean = np.mean(diff_coef, axis=0)
    # diff_coef_std = np.std(diff_coef, axis=0)
    # fig, (ax1, ax2) = plt.subplots(1, 2)

    # ax1.plot(x, y, c=colors[0])
    # ax1.set_aspect('equal')
    # ax1.set_xlabel("Position [m]")
    # ax1.set_ylabel("Position [m]")
    # ax1.minorticks_on()

    # ax2.fill_between(time_array, diff_coef_mean + diff_coef_std,
    #                   diff_coef_mean-diff_coef_std, alpha=0.3, color=colors[1])
    # ax2.plot(time_array, diff_coef_mean, c=colors[1])
    # ax2.axhline(D,c=colors[0])
    # ax2.set_xlabel("Time [s]")
    # ax2.set_ylabel("Diffusion coefficient [m²/s]")
    # ax2.minorticks_on()
    # plt.tight_layout()
    # plt.savefig("figures/simulateur1/numéro2.pdf")
    # plt.show()



    ################
    ### Numéro 3 ###
    ################
    # precision = np.linspace(0, 1e-7, 100)
    # coeff_array = np.empty((30, 100))

    # for j in tqdm(range(30)):
    #     temp_coeff = []
    #     for i in precision:
    #         dt=0.01
    #         D=2.5e-13
    #         t=20
    #         Simulation = DiffusionSimulator(t=t, dt=dt, dx=i, D=D)
    #         time_array = np.linspace(0,Simulation.TotalTime,
    #                                 int(Simulation.TotalTime/Simulation.TimeSteps))
    #         Simulation.generation_motion_rectangle(np.inf)
    #         x,y = Simulation.data

    #         def droite(x,a,b):
    #             return 4*a*x + b

    #         popt, pcov = curve_fit(droite, time_array[:-1],
    #                             Simulation.MeanSquareDisplacement())

    #         temp_coeff.append(popt[0])
    #     coeff_array[j,:] = temp_coeff


    # coeff_mean = np.mean(coeff_array, axis=0)
    # coeff_std = np.std(coeff_array, axis=0)
    # plt.fill_between(precision, coeff_mean+coeff_std,
    #                  coeff_mean-coeff_std, alpha=0.3, color=colors[1])
    # plt.plot(precision, coeff_mean, c=colors[1])
    # plt.minorticks_on()
    # plt.xlabel("Precision of localisation [m]")
    # plt.ylabel("Diffusion coefficient [m²/s]")
    # plt.axhline(D,c=colors[0])
    # plt.tight_layout()
    # plt.savefig("figures/simulateur1/numéro3.pdf")
    # plt.show()



    ################
    ### Numéro 4 ###
    ################
    # dt=0.01
    # D=2.5e-13
    # t=20
    # dx=2e-9
    # Simulation = DiffusionSimulator(t=t, dt=dt, dx=dx, D=D)
    # time_array = np.linspace(0,Simulation.TotalTime,
    #                         int(Simulation.TotalTime/Simulation.TimeSteps))
    # Simulation.generation_motion_rectangle(1e-6)
    # x,y = Simulation.data

    # x_msd = np.empty(x.shape[0])
    # y_msd = np.empty(x.shape[0])
    # for i in range(1,x_msd.shape[0]):
    #     xsquared = np.diff(x[:i])**2
    #     ysquared = np.diff(y[:i])**2

    #     x_msd[i] = np.mean(np.sum(xsquared))
    #     y_msd[i] = np.mean(np.sum(ysquared))

    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # ax1.plot(x, y)
    # ax1.set_xlabel("Position [m]")
    # ax1.set_ylabel("Position [m]")
    # ax1.axvline(1e-6/2, c="k")
    # ax1.axvline(-1e-6/2, c="k")
    # ax1.minorticks_on()

    # ax2.plot(time_array, x_msd, label="x component")
    # ax2.plot(time_array, y_msd, label="y component")
    # ax2.legend()
    # ax2.set_xlabel("Time [s]")
    # ax2.set_ylabel("MSD")
    # ax2.minorticks_on()
    # plt.tight_layout()
    # plt.savefig("figures/simulateur1/numéro4.pdf")
    # plt.show()
