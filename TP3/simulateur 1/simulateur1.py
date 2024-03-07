import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.morphology import disk
from scipy import stats
from scipy.optimize import curve_fit

colors = ['#003D5B', '#D1495B', '#EDAE49', '#00798C', '#401F3E','blue','green','black','red']

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

    def GenerateTrajectories(self, diffusioncoefficient:float, particledensity:float):
        trajectories = {}
        fov_y = self.PixelLines*self.PixelSize
        fov_x = int(self.PixelNumber/self.PixelLines)*self.PixelSize
        if fov_y >= fov_x:
            interest_zone = np.sqrt(2)*fov_y/2
        else:
            interest_zone = np.sqrt(2)*fov_x/2
        N = int(particledensity*(fov_x*fov_y)/(np.pi*(self.PSF)**2))
        for i in tqdm(range(N)):
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
        for i in tqdm(range(self.PixelNumber)):
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
    

# def AutocorrelationFit(data,a,b,c,d):
#     center=10
#     pos = np.dstack(data)
#     gaussienne = stats.multivariate_normal([center, center], [[b,0],[0,d]]).pdf(pos)
#     gaussienne /= np.max(gaussienne)

#     return (a*gaussienne + c).ravel()



def AutocorrelationFit(data, epsilon, psi, diff_coef):
    center=15
    pos = np.dstack(data)
    tau_p = 8e-6
    tau_l = 20 * tau_p
    N = 7
    wo = 2#50e-9
    wz = 3*wo
    delta_x = 1#25e-9
    delta_y = 1#25e-9

    rep_num = 4 * diff_coef * (tau_p * np.abs(epsilon) + tau_l * np.abs(psi))
    denum = 1 + rep_num/(wo**2)
    a = (0.3535 / N) * ((1 + rep_num/(wo**2))**-1) * (np.sqrt(1 + rep_num/(wz**2)))**-1
    b = (np.abs(epsilon) * delta_x / wo)**2 / denum
    d = (np.abs(psi) * delta_y / wo)**2 / denum
    gaussienne = stats.multivariate_normal([center, center], [[b,0],[0,d]],
                                           allow_singular=True).pdf(pos)
    gaussienne /= np.max(gaussienne)

    return (a*gaussienne).ravel()



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






def overall_acf(xy, epsilon, phi):
    """The component of the ACF due to diffusion is the traditional
    correlation function.

    Args:
        epsilon (int): Spatial lag in pixel on the x axis.
        phi (_type_): Spatiale lag in pizel on the y axis.
        N (_type_): Number of particules.
        wo (_type_): Point spread function of the laser.
        wz (_type_): z-axis beam radius.
        tau_p (_type_): Pixel dwell time.
        tau_l (_type_): Interline time.
        D (float): Diffusion coefficient

    Returns:
        float: Traditional correlation function
    """
    N = 1
    D = 1e-12
    wo = 50e-9
    wz = 3 * wo
    tau_p = 8e-6
    tau_l = 20 * tau_p
    # delta_x = 25e-9
    # delta_y = 25e-9

    delta_x, delta_y = xy
    g = (0.3535 / N) * 1/(1 + (4 * D * (tau_p * np.abs(epsilon) + tau_l * np.abs(phi)))/wo**2) * 1/np.sqrt(1 + (4 * D * (tau_p * np.abs(epsilon) + tau_l * np.abs(phi)))/wz**2)

    s = np.exp(-1 * ((np.abs(epsilon) * delta_x / wo)**2 + (np.abs(phi) * delta_y /wo)**2)/(1 + (4 * D * (tau_p * np.abs(epsilon) + tau_l *phi)/(wo**2))))
    return (g * s).ravel()

def fit_2d_eq27(x, y, z, initial_guess):
    # Flatten the input arrays
    x_data_flat = x.ravel()
    y_data_flat = y.ravel()
    z_data_flat = z.ravel()

    popt, pcov = curve_fit(overall_acf, (x_data_flat, y_data_flat), z_data_flat)

    return popt



pixel_dwell_time = 8e-6 # pixel dwell time
D = 1e-12 # coefficient de diffusion
nx = 30
ny = 30
pixels = (nx,ny) # taille de l'image
psf = 50e-9 # tache du fluorophore / taille du faisceau laser
pixel_size = 25e-9
density = 1



print('Generating trajectories...')
simulator = ICSSimulator(pixels[0]*pixels[1], pixels[1], psf, pixel_size, pixel_dwell_time)
simulator.GenerateTrajectories(D,density)
print('Generating images...')
simulator.GenerateImage(10,False)
image_set = simulator.Images


corr_image = SpatialAutocorrelation(image_set)

image = image_set
autocorr = SpatialAutocorrelation(image_set)

fig = plt.figure(figsize=(10,4))
(ax1, ax2, ax3, ax4, ax5) = fig.subfigures(1, 5)
ax1 = ax1.subplots()
ax2 = ax2.subplots(subplot_kw=dict(projection='3d'))
ax3 = ax3.subplots()
ax4 = ax4.subplots()
ax5 = ax5.subplots()
ax1.set_ylim(image.shape[1], 0)
ax1.set_title('a) Image résultante')
ax1.imshow(image,cmap='bone')
ax1.axis('off')
x = np.arange(image.shape[0])
y = np.arange(image.shape[0])
x,y = np.meshgrid(x,y)

ax2.plot_surface(x,y,autocorr, cmap = 'cividis')
ax2.set_title('B) Autocorrelation function')
ax2.set_xlabel(r'$\xi$')
ax2.set_ylabel(r'$\nu$')
ax2.set_zlabel(r'G($\xi$,$\nu$)')

# popt = fit_2d_eq27(x, y, autocorr, [7, 50e-9, 150e-9, 10e-6, 10e-6, 25e-9, 25e-9, 1e-12])

# ax3.plot(autocorr.ravel())
# ax3.plot(overall_acf((x,y), *popt))


# ax4.imshow(autocorr, cmap="magma")
# ax5.imshow(overall_acf((x,y), *popt).reshape((20,20)))
# plt.show()





(epsilon, psi, diff_coeff), pcov = curve_fit(AutocorrelationFit,
                                        (x,y),
                                        autocorr.ravel(),
                                        p0=[1, 1, 3e-12],
                                        maxfev=10000,
                                        bounds=(0, 20))


print(epsilon, psi, diff_coeff)

ax3.plot(autocorr.ravel(), label="autocorr")
ax3.plot(AutocorrelationFit((x,y),epsilon, psi, diff_coeff), label="Fit")
ax3.legend()

ax4.imshow(autocorr, cmap="magma")

fit_image =AutocorrelationFit((x,y),epsilon, psi, diff_coeff).reshape(image.shape)

ax5.imshow(fit_image)
plt.show()


fit_axis_x = fit_image[fit_image.shape[0]//2,fit_image.shape[0]//2:]
fit_axis_y = fit_image[fit_image.shape[0]//2:,fit_image.shape[0]//2]


plt.plot(fit_axis_x)
plt.plot(fit_axis_y)
plt.show()
