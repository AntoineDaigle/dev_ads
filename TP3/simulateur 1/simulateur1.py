import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.morphology import disk
from scipy import stats
from scipy.optimize import curve_fit
import json

colors = ['#005C69', '#023618', '#E9B44C', '#B0413E', '#83C5BE']

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

class Image_Simulator():

    def __init__(self, pixel_size, pixel_numberx, pixel_numbery):
        self.pixel_size = pixel_size
        self.pixel_x = pixel_numberx
        self.pixel_y = pixel_numbery
        self.image = np.zeros((pixel_numberx, pixel_numbery))
    
    def emetteur(self, spot_size, center, N_photon):  
        def gaussian2D(x0,y0,s):
            x, y = np.meshgrid(np.arange(-self.pixel_x/2, self.pixel_x/2), np.arange(-self.pixel_y/2, self.pixel_y/2))
            pos = np.dstack((x,y))
            gaussienne = stats.multivariate_normal([x0,y0], [[s**2,0],[0,s**2]]).pdf(pos)
            if np.nansum(gaussienne) == 0:
                return gaussienne
            return (N_photon * gaussienne/np.nansum(gaussienne))
        x0, y0 = center
        x0 /= self.pixel_size
        y0 /= self.pixel_size
        spot_size /= self.pixel_size
        grid = gaussian2D(x0,y0,spot_size)
        grid = np.transpose(grid)
        self.image += grid

    def photon_counting(self):
        def shotnoise(a):
            return np.random.poisson(a)
        sn = np.vectorize(shotnoise)
        self.image = sn(self.image)

    def background_noise(self, b):
        for i in range(self.pixel_x):
            for j in range(self.pixel_y): 
                self.image[i,j] += np.abs(np.random.normal(self.image[i,j], b))

class DiffusionSimulator():

    def __init__(self, dt:float, D:float, center:tuple, iterations:int):
        """Class that simulate de diffusion of a set of particule.

        Args:
            t (float): Total time of the simulation in seconds.
            dt (float): Time increment in second.
            D (float): Diffusion coefficient.
            Center (tuple): Initial position of the particle
        """
        self.Iterations = iterations
        self.TimeSteps = dt
        self.DiffusionCoefficient = D
        self.StartingPoint = center

    def generation_motion(self, radius):
        """
        Function to generate motion in a rectangle

        xlim: extent of the zone in x
        ylim: extent of the zone in y
        """
        x, y = np.zeros(self.Iterations), np.zeros(self.Iterations)
        old_x, old_y = self.StartingPoint
        x[0],y[0] = old_x,old_y
        factor = np.sqrt(2*self.DiffusionCoefficient*self.TimeSteps)
        for i in range(1,self.Iterations):
            r1 =  np.random.normal(0,factor) 
            r2 =  np.random.normal(0,factor) 
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

class ICSSimulator():

    def __init__(self, pixelnumber:int, pixellines:int, psfsize:float, 
                 pixelsize:float, pixeldwelltime:float, images = 1):
        self.PixelNumber = pixelnumber
        self.PixelLines = pixellines
        self.PixelSize = pixelsize
        self.PixelDwellTime = pixeldwelltime
        self.PSF = psfsize
        self.ImageNumber = images

    def GenerateTrajectories(self, diffusioncoefficient:float,
                             particledensity:float):
        trajectories = {}
        fov_y = self.PixelLines*self.PixelSize
        fov_x = int(self.PixelNumber/self.PixelLines)*self.PixelSize
        if fov_y >= fov_x:
            interest_zone = np.sqrt(2)*fov_y/2
        else:
            interest_zone = np.sqrt(2)*fov_x/2
        N = int(particledensity*(fov_x*fov_y)/(np.pi*(self.PSF)**2))
        # for i in tqdm(range(N)):
        for i in range(N):
            center = (np.inf, np.inf)
            while np.abs(center[1])>fov_x/2 or np.abs(center[0])>fov_y/2:
                center = np.random.uniform(-interest_zone,interest_zone,2)
            Simulator = DiffusionSimulator(self.PixelDwellTime,diffusioncoefficient,center,self.ImageNumber*self.PixelNumber)
            Simulator.generation_motion(interest_zone)
            trajectory = Simulator.data
            trajectories[f'{i}'] = trajectory
        self.Trajectories = trajectories

    def GenerateImage(self, intensity, shotnoise=False):
        pixel_x = int(self.PixelNumber/self.PixelLines)
        pixel_y = int(self.PixelLines)
        image = np.empty((pixel_x, pixel_y))
        # for i in tqdm(range(self.PixelNumber)):
        for i in range(self.PixelNumber):
            sim = Image_Simulator(self.PixelSize, pixel_x, pixel_y)
            n = i // pixel_x
            m = i % pixel_x
            for particle in self.Trajectories.keys():
                position = (self.Trajectories[particle][1][i],self.Trajectories[particle][0][i])
                if np.abs(position[0]) < (pixel_x-1)/2*pixel_size or np.abs(position[1]) < (pixel_y-1)/2*pixel_size:
                    sim.emetteur(psf/2, position, intensity)
            if shotnoise:
                sim.photon_counting()
            set = sim.image
            image[m,n] = set[m,n]
        image = np.transpose(image)

        image -= np.min(image)
        self.Images = (255 * image / np.max(image)).astype(np.uint8)
        return np.transpose(set)




def GaussianFit(data,a,b,c,d):
    center = 10
    pos = np.dstack(data)
    gaussienne = stats.multivariate_normal([center, center],
                                           [[b,0], [0,d]]).pdf(pos)
    gaussienne /= np.max(gaussienne)

    return (a*gaussienne + c).ravel()



# def AutocorrelationFit(data, epsilon, psi, diff_coef):
#     center=10
#     pos = np.dstack(data)
#     tau_p = 8e-6
#     tau_l = 2 * center * tau_p
#     N = 74
#     wo = 2#50e-9
#     wz = 3*wo
#     delta_x = 1#25e-9
#     delta_y = 1#25e-9

#     rep_num = 4 * diff_coef * (tau_p * np.abs(epsilon) + tau_l * np.abs(psi))
#     denum = 1 + rep_num/(wo**2)
#     a = (0.3535 / N) * ((1 + rep_num/(wo**2))**-1) * (np.sqrt(1 + rep_num/(wz**2)))**-1
#     b = (np.abs(epsilon) * delta_x / wo)**2 / denum
#     d = (np.abs(psi) * delta_y / wo)**2 / denum
#     gaussienne = stats.multivariate_normal([center, center], [[b,0],[0,d]],
#                                            allow_singular=True).pdf(pos)
#     gaussienne /= np.max(gaussienne)

#     return (a*gaussienne).ravel()

# def AutocorrelationFit(data, epsilon, psi, diff_coef):
#     center=10
#     pos = np.dstack(data)
#     tau_p = 8e-6
#     tau_l = 2 * center * tau_p
#     N = 1
#     wo = 2#50e-9
#     wz = 3*wo
#     delta_x = 1#25e-9
#     delta_y = 1#25e-9

#     rep_num = 4 * diff_coef * (tau_p * np.abs(epsilon) + tau_l * np.abs(psi))
#     a = (0.3535 / N) * ((1 + rep_num/(wo**2))**-1) ** 3/2
#     gaussienne = stats.multivariate_normal([center, center], [[b,0],[0,d]],
#                                            allow_singular=True).pdf(pos)
#     gaussienne /= np.max(gaussienne)

#     return (a*gaussienne).ravel()



# def AutocorrelationFit(data, a, b, c, d):
#     center=10
#     pos = np.dstack(data)
#     tau_p = 8e-6
#     tau_l = 2 * center * tau_p
#     N = 7
#     wo = 2#50e-9
#     wz = 3*wo
#     delta_x = 1#25e-9
#     delta_y = 1#25e-9

#     # rep_num = 4 * diff_coef * (tau_p * np.abs(epsilon) + tau_l * np.abs(psi))
#     # denum = 1 + rep_num/(wo**2)
#     # a = (0.3535 / N) * ((1 + rep_num/(wo**2))**-1) * (np.sqrt(1 + rep_num/(wz**2)))**-1
#     # b = (np.abs(epsilon) * delta_x / wo)**2 / denum
#     # d = (np.abs(psi) * delta_y / wo)**2 / denum
#     gaussienne = stats.multivariate_normal([center, center], [[b,0],[0,d]],
#                                            allow_singular=True).pdf(pos)
#     gaussienne /= np.max(gaussienne)

#     return (a*gaussienne + c).ravel()



def SpatialAutocorrelation(image):
    """Fonction pour que antoine comprenne mon code  (tu le comprendra pas mon esti)
    There is still a factor 10 in the results (about 10)
    Args:
        image (array): image to do the autocorrelation of
    Returns:
        autocorr (array): array of image with the autocorrelation function following equation 7 from the paper
    """
    shape = image.shape # Calculating the shape of the image
    image = image.ravel() # Flattening the image so it is a 1D array
    mean_image = np.mean(image) # Calculating the mean pixel intensity
    normalized_image = image - mean_image # Normalizing following equation 2
    signal_1, signal_2 = normalized_image, normalized_image # assigning the normalized image to both signal 1 and signal 2
    autocorr = [] #creating a list to put our autocorrelation values in
    for _ in range(image.shape[0]): #iterating for the shape of the array
        autocorr.append(np.average(signal_1*signal_2)/mean_image**2) # Equation 7
        signal_1 = np.roll(signal_1, 1) # Moving signal_1
    autocorr = np.array(autocorr) # transforming autocorrelation in array for futur operations
    return  np.fft.ifftshift(autocorr.reshape(shape)) # reshaping into image shape and shifting to bring the peaks at the center

def g_function(data, diff_coeff, N):
    tau = 50e-6
    wo=200e-9
    fov_y = ny*pixel_size
    fov_x = nx*pixel_size
    # N = int(density*(fov_x*fov_y)/(np.pi*(psf)**2))
    g = (0.3535 / N) * 1/((1 + 4 * diff_coeff * tau * data / np.square(wo))**3/2)
    return g

####################
### Simulation A ###
####################

# data = {}

# for j in tqdm([0.1, 1, 10]):
#     for k in tqdm(range(50)):
        # pixel_dwell_time = 50e-6 # pixel dwell time
        # D = 1e-12 # coefficient de diffusion
        # nx = 52
        # ny = 52
        # pixels = (nx,ny) # taille de l'image
        # psf = 200e-9 # tache du fluorophore / taille du faisceau laser
        # pixel_size = 100e-9
        # # density = 1

        # # print('Generating trajectories...')
        # simulator = ICSSimulator(pixels[0]*pixels[1], pixels[1], psf, pixel_size,
        #                         pixel_dwell_time)
        # simulator.GenerateTrajectories(D,j)
        # # print('Generating images...')
        # simulator.GenerateImage(10,False)
        # image_set = simulator.Images
        # image_set = image_set[1:-1,1:-1]


        # autocorr = SpatialAutocorrelation(image_set)


        # signal = autocorr[autocorr.shape[0]//2,autocorr.shape[0]//2:]

        # (d_c, n), pcov = curve_fit(g_function,
        #                         np.arange(25),
        #                         signal,
        #                         p0=[1e-12, 21],
        #                         maxfev=10000,
        #                         bounds=(0, 1000))
        # with open("data/question_1/data.txt", "a") as f:
        #     f.write(f"density_{j}_{k}\t{d_c}\n")







####################
### Simulation B ###
####################

# data = {}

# for j in tqdm([0.1e-12, 1e-12, 10e-12]):
#     for k in tqdm(range(5)):
#         pixel_dwell_time = 50e-6 # pixel dwell time
#         # D = 1e-12 # coefficient de diffusion
#         nx = 52
#         ny = 52
#         pixels = (nx,ny) # taille de l'image
#         psf = 200e-9 # tache du fluorophore / taille du faisceau laser
#         pixel_size = 100e-9
#         density = 0.1

#         # print('Generating trajectories...')
#         simulator = ICSSimulator(pixels[0]*pixels[1], pixels[1], psf, pixel_size,
#                                 pixel_dwell_time)
#         simulator.GenerateTrajectories(j,density)
#         # print('Generating images...')
#         simulator.GenerateImage(10,False)
#         image_set = simulator.Images
#         image_set = image_set[1:-1,1:-1]


#         autocorr = SpatialAutocorrelation(image_set)


#         signal = autocorr[autocorr.shape[0]//2,autocorr.shape[0]//2:]
#         signal[5:] = 0


#         (d_c, n), pcov = curve_fit(g_function,
#                                 np.arange(25),
#                                 signal,
#                                 p0=[1e-12, 21],
#                                 maxfev=10000,
#                                 bounds=(0, 1000))
#         with open("data/question_2/data.txt", "a") as f:
#             f.write(f"diff_coeff_{j}_{k}\t{d_c}\n")






####################
### Simulation C ###
####################

data = {}

for j in tqdm([50e-9, 50e-2, 50e-3]):
    for k in tqdm(range(2)):
        # pixel_dwell_time = 50e-6 # pixel dwell time
        D = 1e-12 # coefficient de diffusion
        nx = 52
        ny = 52
        pixels = (nx,ny) # taille de l'image
        psf = 200e-9 # tache du fluorophore / taille du faisceau laser
        pixel_size = 100e-9
        density = 1

        # print('Generating trajectories...')
        simulator = ICSSimulator(pixels[0]*pixels[1], pixels[1], psf,
                                 pixel_size, j)
        simulator.GenerateTrajectories(D,density)
        # print('Generating images...')
        simulator.GenerateImage(10,False)
        image_set = simulator.Images
        image_set = image_set[1:-1,1:-1]


        autocorr = SpatialAutocorrelation(image_set)


        signal = autocorr[autocorr.shape[0]//2,autocorr.shape[0]//2:]
        signal[5:] = 0
        # plt.plot(signal)


        (d_c, n), pcov = curve_fit(g_function,
                                np.arange(25),
                                signal,
                                p0=[1e-12, 21],
                                maxfev=10000,
                                bounds=(0, 1000))
        # plt.plot(np.arange(25), g_function(np.arange(25), d_c, n))
        # plt.show()
        with open("data/question_3/data.txt", "a") as f:
            f.write(f"taup_{j}_{k}\t{d_c}\n")











# pixel_dwell_time = 50e-6 # pixel dwell time
# D = 1e-12 # coefficient de diffusion
# nx = 52
# ny = 52
# pixels = (nx,ny) # taille de l'image
# psf = 200e-9 # tache du fluorophore / taille du faisceau laser
# pixel_size = 100e-9
# density = 0.1

# # print('Generating trajectories...')
# simulator = ICSSimulator(pixels[0]*pixels[1], pixels[1], psf, pixel_size,
#                         pixel_dwell_time)
# simulator.GenerateTrajectories(D,density)
# # print('Generating images...')
# simulator.GenerateImage(10,False)
# image_set = simulator.Images
# image_set = image_set[1:-1,1:-1]


# autocorr = SpatialAutocorrelation(image_set)


# signal = autocorr[autocorr.shape[0]//2,autocorr.shape[0]//2:]
# signal[5:] = 0

# (d_c, n), pcov = curve_fit(g_function,
#                         np.arange(25),
#                         signal,
#                         p0=[1e-12, 21],
#                         maxfev=10000,
#                         bounds=(0, 1000))

# print(d_c, n)

# fig = plt.figure(figsize=(10,4))
# (ax1, ax2, ax4, ax5, ax6) = fig.subfigures(1, 5)
# ax1 = ax1.subplots()
# ax2 = ax2.subplots(subplot_kw=dict(projection='3d'))
# ax4 = ax4.subplots()
# ax5 = ax5.subplots()
# ax6 = ax6.subplots()
# ax1.set_ylim(image_set.shape[1], 0)
# ax1.set_title('a) Image résultante')
# ax1.imshow(image_set,cmap='bone')
# ax1.axis('off')
# x = np.arange(image_set.shape[0])
# y = np.arange(image_set.shape[0])
# x,y = np.meshgrid(x,y)

# ax2.plot_surface(x,y,autocorr, cmap = 'cividis')
# ax2.set_title('B) Autocorrelation function')
# ax2.set_xlabel(r'$\xi$')
# ax2.set_ylabel(r'$\nu$')
# ax2.set_zlabel(r'G($\xi$,$\nu$)')

# ax4.imshow(autocorr)
# fit_image = autocorr

# (a, b, c, d), pcov = curve_fit(GaussianFit,
#                                 (x,y),
#                                 autocorr.ravel(),
#                                 p0=[1, 1, 1, 1],
#                                 maxfev=10000,
#                                 bounds=(0, 20))

# print(a, b, c, d)

# (epsilon, psi, diff_coeff), pcov = curve_fit(AutocorrelationFit,
#                                                 (x,y),
#                                                 autocorr.ravel(),
#                                                 p0=[1, 1, 1e-12],
#                                                 maxfev=10000,
#                                                 bounds=(0, 20))


# print(f"x lag: {epsilon}\ny lag: {psi}\nDiffusion coeff: {diff_coeff}")


# ax5.plot(signal)
# ax5.plot(np.arange(25), g_function(np.arange(25), d_c, n))




# fit_image =AutocorrelationFit((x,y), epsilon, psi, diff_coeff).reshape(image_set.shape)
# fit_image =AutocorrelationFit((x,y),epsilon, psi, diff_coeff).reshape(image_set.shape)
# fit_image =GaussianFit((x,y),a, b, c, d).reshape(image_set.shape)

# ax5.imshow(fit_image)


# fit_axis_x = fit_image[fit_image.shape[0]//2,fit_image.shape[0]//2:]
# fit_axis_y = fit_image[fit_image.shape[0]//2:,fit_image.shape[0]//2]


# ax6.plot(fit_axis_x)
# ax6.plot(fit_axis_y)
plt.show()