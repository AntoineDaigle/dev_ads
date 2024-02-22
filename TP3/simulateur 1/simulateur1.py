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
        return np.transpose(set)
    
def SpatialAutocorrelation(image):
    FFT =np.fft.fft2(image)
    FFTstar = np.conjugate(FFT)
    upperterm = np.abs(np.fft.fftshift(np.fft.ifft2(FFT*FFTstar)))
    lowerterm = np.average(image)**2
    autocorr = upperterm/lowerterm - 1
    return  autocorr




dt = 10e-6 # pixel dwell time
D = 1e-12 # coefficient de diffusion
pixels = (50,10) # taille de l'image
psf = 50e-9 # tache du fluorophre / taille du faisceau laser
ps = 25e-9 # 
time=2
densitys = [0.02]

for density in densitys:
    print(density)
    print('Generating trajectories...')
    simulator = ICSSimulator(pixels[0]*pixels[1], pixels[1], psf, ps, dt)
    simulator.GenerateTrajectories(D,density)
    trajectories = simulator.Trajectories
    print('Generating images...')
    set = simulator.GenerateImage(10,False)
    image_set = simulator.Images

    np.save(f'simulateur 1/data/image_{dt}s_{D}_{psf}_{ps}_{density}.npy', image_set)

fig, ax1 = plt.subplots(1,1, figsize=(10,5))
ax1.imshow(image_set, cmap = 'magma')
# ax2.imshow(set, cmap = 'magma')
# for particle in trajectories.keys():
#     trajectory = trajectories[particle]
#     ax2.plot(trajectory[0], trajectory[1])
# ax2.vlines([-pixels[1]/2*ps, pixels[1]/2*ps], -pixels[0]/2*ps, pixels[0]/2*ps, color = 'k')
# ax2.hlines([-pixels[0]/2*ps, pixels[0]/2*ps], -pixels[1]/2*ps, pixels[1]/2*ps, color = 'k')
plt.show()

# image = image_set

# fns = ['simulateur 1\data\image_1e-07s_1e-12_5e-08_2.5e-08_0.1.npy',
#        'simulateur 1\data\image_1e-07s_1e-12_5e-08_2.5e-08_0.25.npy', 
#       "simulateur 1\data\image_1e-07s_1e-12_5e-08_2.5e-08_0.5.npy",
#       "simulateur 1\data\image_1e-07s_1e-12_5e-08_2.5e-08_1.npy", 
#       'simulateur 1\data\image_1e-07s_1e-12_5e-08_2.5e-08_2.5.npy', 
#       "simulateur 1\data\image_1e-07s_1e-12_5e-08_2.5e-08_5.npy",
#       'simulateur 1\data\image_1e-07s_1e-12_5e-08_2.5e-08_10.npy']

# for fn in fns:
#     image = np.load(fn)

#     fig = plt.figure(figsize=(11,5),layout='tight')
#     (ax1,ax2) = fig.subfigures(1,2)
#     ax2 = ax2.subplots(subplot_kw=dict(projection='3d'))
#     ax1 = ax1.subplots()
#     ax1.set_ylim(image.shape[1], 0)
#     ax1.set_title('a) Image résultante')
#     ax1.imshow(image,cmap='bone')
#     ax1.axis('off')

#     autocorr = SpatialAutocorrelation(image)


#     x = np.arange(image.shape[0])
#     y = np.arange(image.shape[0])
#     x,y = np.meshgrid(x,y)
#     ax2.plot_surface(x,y,autocorr, cmap = 'cividis')
#     ax2.set_title('B) Autocorrelation function')
#     ax2.set_xlabel(r'$\xi$')
#     ax2.set_ylabel(r'$\nu$')
#     ax2.set_zlabel(r'G($\xi$,$\nu$)')
#     plt.savefig('simulateur 1/figures/fig_{fn}.svg')
#     plt.show()
