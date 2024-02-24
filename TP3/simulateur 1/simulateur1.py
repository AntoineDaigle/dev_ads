import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.morphology import disk
from scipy import stats

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
        N = int(particledensity*(fov_x*fov_y)/(np.pi*(self.PSF/2)**2))
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
                if np.abs(position[0]) < (pixel_x-1)/2*ps or np.abs(position[1]) < (pixel_y-1)/2*ps:
                    sim.emetteur(psf, position, intensity)
            if shotnoise:
                sim.photon_counting()
            set = sim.image
            image[m,n] = set[m,n]
        self.Images = np.transpose(image)
        # return np.transpose(set)
    
def SpatialAutocorrelation(image):
    FFT =np.fft.fft2(image)
    FFTstar = np.conjugate(FFT)
    upperterm = np.abs(np.fft.fftshift(np.fft.ifft2(FFT*FFTstar)))
    lowerterm = np.average(image)**2
    autocorr = upperterm/lowerterm - 1
    return  autocorr

def G(epsilon, phi, N, wo, wz, tau_p, tau_l, gamma=0.3535):
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
        gamma (float, optional): Shape factor due to uneven illumination. Defaults to 0.3535.

    Returns:
        float: Traditional correlation function
    """
    return (gamma / N) * 1/(1 + (4 * D * (tau_p * np.abs(epsilon) + tau_l * np.abs(phi)))/wo**2) * 1/np.sqrt(1 + (4 * D * (tau_p * np.abs(epsilon) + tau_l * np.abs(phi)))/wz**2)

def S(epsilon, phi, D, delta_x, delta_y, tau_l, tau_p, wo):
    """Correlation function decays due to movement of the laser beam scanning.

    Args:
        epsilon (_type_): Spatial lag in x
        phi (_type_): Spatial lag in y
        D (_type_): Diffusion coeff
        delta_x (_type_): Pixel size in x
        delta_y (_type_): Pixel size in y
        tau_l (_type_): Interline time in y
        tau_p (_type_): Pixel dwell time in x
        wo (_type_): Point spread function of the laser beam.

    Returns:
        _type_: _description_
    """
    return np.exp(-1 * ((np.abs(epsilon) * delta_x / wo)**2 + (np.abs(phi) * delta_y /wo)**2)/(1 + (4 * D * (tau_p * np.abs(epsilon) + tau_l *phi)/(wo**2))))






dt = 10e-6 # pixel dwell time
D = 1e-12 # coefficient de diffusion
pixels = (50,10) # taille de l'image
psf = 50e-9 # tache du fluorophre / taille du faisceau laser
ps = 25e-9 # 
time=2
density = 0.02

print('Generating trajectories...')
simulator = ICSSimulator(pixels[0]*pixels[1], pixels[1], psf, ps, dt)
simulator.GenerateTrajectories(D,density)
print('Generating images...')
simulator.GenerateImage(10,False)
image_set = simulator.Images

# np.save(f'simulateur 1/data/image_{dt}s_{D}_{psf}_{ps}_{density}.npy', image_set)

fig, ax1 = plt.subplots(1,1, figsize=(10,5))
ax1.imshow(image_set, cmap = 'magma')
plt.show()
