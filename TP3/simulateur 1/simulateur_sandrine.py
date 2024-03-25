import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats
from scipy.optimize import curve_fit
from scipy.ndimage import correlate


colors = ['#003D5B', '#D1495B', '#EDAE49', '#00798C', '#401F3E','blue','green','black','red']

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

class Image_Simulator():
    """
    Class to simulate a frame given by diffusing particles

    Args:
    pixel_size (float): size of the pixels of the image
    pixel_numberx (int): number of pixels in x
    pixel_numbery (int): number of pixels in y
    """

    def __init__(self, pixel_size, pixel_numberx, pixel_numbery):
        self.pixel_size = pixel_size
        self.pixel_x = pixel_numberx
        self.pixel_y = pixel_numbery
        self.image = np.zeros((pixel_numberx, pixel_numbery))
    
    def emetteur(self, spot_size, center, N_photon):  
        """method to add an emettor in the image

        Args:
            spot_size (float): spot size in m
            center (float): center in m
            N_photon (number of photons of the ): _description_
        """
        def gaussian2D(x0,y0,s):
            x, y = np.meshgrid(np.arange(-self.pixel_x/2, self.pixel_x/2), np.arange(-self.pixel_y/2, self.pixel_y/2))
            pos = np.dstack((x,y))
            gaussienne = stats.multivariate_normal([x0,y0], [[s**2,0],[0,s**2]]).pdf(pos)
            if np.nansum(gaussienne) == 0:
                return N_photon * gaussienne
            return (N_photon * gaussienne/np.max(gaussienne))
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

    def GenerateTrajectories(self, diffusioncoefficient:float, particledensity:float):
        trajectories = {}
        fov_y = self.PixelLines*self.PixelSize
        fov_x = int(self.PixelNumber/self.PixelLines)*self.PixelSize
        if fov_y >= fov_x:
            interest_zone = np.sqrt(2)*fov_y/2
        else:
            interest_zone = np.sqrt(2)*fov_x/2

        N = int(particledensity*(fov_x*fov_y)/(np.pi*(self.PSF)**2)/3) ## Changed this
        
        for i in range(N): #tqdm here
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
        for i in range(self.PixelNumber): #tqdm here
            sim = Image_Simulator(self.PixelSize, pixel_x, pixel_y)
            n = i // pixel_x
            m = i % pixel_x
            for particle in self.Trajectories.keys():
                position = (self.Trajectories[particle][0][i],self.Trajectories[particle][1][i])
                # if np.abs(position[0]) < (pixel_x)/2*ps or np.abs(position[1]) < (pixel_y)/2*ps:
                sim.emetteur(psf, position, intensity) # changed this
            if shotnoise:
                sim.photon_counting()
            set = sim.image
            image[m,n] = set[m,n]
        image = np.transpose(image)

        #converting to 8 bit image
        image -= np.min(image)
        self.Images = (255*image/np.max(image)).astype(np.uint8) # ajouté des trucs tantot
        return np.transpose(set)
    
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
    mean_image = np.average(image) # Calculating the mean pixel intensity
    normalized_image = image - mean_image # Normalizing following equation 2
    signal_1, signal_2 = normalized_image, normalized_image # assigning the normalized image to both signal 1 and signal 2 
    autocorr = [] #creating a list to put our autocorrelation values in
    for _ in range(image.shape[0]): #iterating for the shape of the array
        autocorr.append(np.average(signal_1*signal_2)/mean_image**2) # Equation 7
        signal_1 = np.roll(signal_1, 1) # Moving signal_1
    autocorr = np.array(autocorr) # transforming autocorrelation in array for futur operations
    return  np.fft.ifftshift(autocorr.reshape(shape)) # reshaping into image shape and shifting to bring the peaks at the center

def AutocorrelationFit(data,a,b,c):
    center=data.shape[1]/2
    # center = 10
    pos = np.dstack(data)
    gaussienne = stats.multivariate_normal([center, center], [[b**2,0],[0,b**2]]).pdf(pos)
    gaussienne /= np.max(gaussienne)
    return (a*gaussienne + c).ravel()

def g_function(data, diff_coeff, N):
    tau = 50e-6
    wo=200e-9
    # fov_y = pixels[0]*ps
    # fov_x = pixels[1]*ps
    # N = int(density*(fov_x*fov_y)/(np.pi*(psf)**2))
    g = (0.3535 / N) * 1/((1 + 4 * diff_coeff * tau * data / np.square(wo))**3/2)
    return g


##################
### Question A ###
##################

# dt = 50e-6 # pixel dwell time
# D = 1e-12 # coefficient de diffusion
# pixels = (50,50) # taille de l'image
# psf = 200e-9 # tache du fluorophre / taille du faisceau laser
# # pss = [450e-9,500e-9,600e-9] # taille des pixels de l'image
# ps = 100e-9
# density = 10
#  # densité de particule / faisceau laser
# intensity = 500


# for j in tqdm([0.1, 1, 10]):
#     for k in tqdm(range(50)):
#         simulator = ICSSimulator(pixels[0]*pixels[1], pixels[1], psf, ps, dt)
#         simulator.GenerateTrajectories(D,j)
#         trajectories = simulator.Trajectories
#         simulator.GenerateImage(intensity,True)
#         image = simulator.Images

#         autocorr = SpatialAutocorrelation(image)


#         signal = autocorr[autocorr.shape[0]//2,autocorr.shape[0]//2:]

#         (d_c, n), pcov = curve_fit(g_function,
#                                 np.arange(25),
#                                 signal,
#                                 p0=[1e-12, 21],
#                                 maxfev=10000,
#                                 bounds=(0, 1000))

#         with open("data/question_1/data.txt", "a") as f:
#             f.write(f"density_{j}_{k}\t{d_c}\n")

##################
### Question B ###
##################


# dt = 50e-6 # pixel dwell time
# # D = 1e-12 # coefficient de diffusion
# pixels = (50,50) # taille de l'image
# psf = 200e-9 # tache du fluorophre / taille du faisceau laser
# # pss = [450e-9,500e-9,600e-9] # taille des pixels de l'image
# ps = 100e-9
# density = 0.1
#  # densité de particule / faisceau laser
# intensity = 500


# for j in tqdm([0.1e-12, 1e-12, 10e-12]):
#     for k in tqdm(range(50)):
#         simulator = ICSSimulator(pixels[0]*pixels[1], pixels[1], psf, ps, dt)
#         simulator.GenerateTrajectories(j,density)
#         trajectories = simulator.Trajectories
#         simulator.GenerateImage(intensity,True)
#         image = simulator.Images

#         autocorr = SpatialAutocorrelation(image)


#         signal = autocorr[autocorr.shape[0]//2,autocorr.shape[0]//2:]

#         (d_c, n), pcov = curve_fit(g_function,
#                                 np.arange(25),
#                                 signal,
#                                 p0=[1e-12, 21],
#                                 maxfev=10000,
#                                 bounds=(0, 1000))

#         with open("data/question_2/data.txt", "a") as f:
#             f.write(f"diff_coeff_{j}_{k}\t{d_c}\n")




##################
### Question C ###
##################


# # dt = 50e-6 # pixel dwell time
# D = 1e-12 # coefficient de diffusion
# pixels = (50,50) # taille de l'image
# psf = 200e-9 # tache du fluorophre / taille du faisceau laser
# # pss = [450e-9,500e-9,600e-9] # taille des pixels de l'image
# ps = 100e-9
# density = 0.1
#  # densité de particule / faisceau laser
# intensity = 500


# for j in tqdm([50e-9, 50e-6, 50e-3]):
    
#     def g_function(data, diff_coeff, N):
#         tau = j
#         print(tau)
#         wo=200e-9
#         g = (0.3535 / N) * 1/((1 + 4 * diff_coeff * tau * data / np.square(wo))**3/2)
#         return g
#     for k in tqdm(range(1)):
#         simulator = ICSSimulator(pixels[0]*pixels[1], pixels[1], psf, ps, j)
#         simulator.GenerateTrajectories(D,density)
#         trajectories = simulator.Trajectories
#         simulator.GenerateImage(intensity,True)
#         image = simulator.Images

#         autocorr = SpatialAutocorrelation(image)

        


#         signal = autocorr[autocorr.shape[0]//2,autocorr.shape[0]//2:]

#         (d_c, n), pcov = curve_fit(g_function,
#                                 np.arange(25),
#                                 signal,
#                                 p0=[1e-12, 21],
#                                 maxfev=10000,
#                                 bounds=(0, 1000))
        
#         fig, (ax0, ax1, ax2) = plt.subplots(3)

#         ax0.imshow(image)
#         ax1.imshow(autocorr)
#         ax2.plot(signal)
#         ax2.plot(np.arange(25), g_function(np.arange(25), d_c, n))
#         plt.show()

#         with open("data/question_3/data.txt", "a") as f:
#             f.write(f"dwelltime_{j}_{k}\t{d_c}\n")




##################
### Question C ###
##################


dt = 50e-9 # pixel dwell time
# D = 1e-12 # coefficient de diffusion
pixels = (50,50) # taille de l'image
psf = 200e-9 # tache du fluorophre / taille du faisceau laser
# pss = [450e-9,500e-9,600e-9] # taille des pixels de l'image
ps = 100e-9
density = 0.1
 # densité de particule / faisceau laser
intensity = 500


for j in tqdm(np.linspace(0.1e-12, 100e-12)):
    
    def g_function(data, diff_coeff, N):
        tau = 50e-9
        wo=200e-9
        g = (0.3535 / N) * 1/((1 + 4 * diff_coeff * tau * data / np.square(wo))**3/2)
        return g
    

    for k in tqdm(range(10)):
        simulator = ICSSimulator(pixels[0]*pixels[1], pixels[1], psf, ps, dt)
        simulator.GenerateTrajectories(j,density)
        trajectories = simulator.Trajectories
        simulator.GenerateImage(intensity,True)
        image = simulator.Images

        autocorr = SpatialAutocorrelation(image)

        


        signal = autocorr[autocorr.shape[0]//2,autocorr.shape[0]//2:]

        (d_c, n), pcov = curve_fit(g_function,
                                np.arange(25),
                                signal,
                                p0=[1e-12, 21],
                                maxfev=10000,
                                bounds=(0, 1000))
        with open("data/question_4/data.txt", "a") as f:
            f.write(f"dwell_diff_{j}_{k}\t{d_c}\n")

