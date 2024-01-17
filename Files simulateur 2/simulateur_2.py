import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import matplotlib.animation as animation
from tqdm import tqdm
from skimage.morphology import disk

colors = ['#003D5B', '#D1495B', '#EDAE49', '#00798C', '#401F3E','blue','green','black','red']

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

class DiffusionSimulator():

    def __init__(self, t:float, dt:float, D:float, center:tuple):
        """Class that simulate de diffusion of a set of particule.

        Args:
            t (float): Total time of the simulation in seconds.
            dt (float): Time increment in second.
            dx (float): Localisation precision in meters.
            D (float): Diffusion coefficient.
        """
        self.TotalTime = t
        self.TimeSteps = dt
        self.DiffusionCoefficient = D
        self.StartingPoint = center


    def generation_motion_circle(self, radius = np.inf):
        n = int(self.TotalTime/self.TimeSteps)
        x, y = np.zeros(n), np.zeros(n)
        old_x, old_y = self.StartingPoint
        x[0],y[0] = old_x,old_y
        factor = np.sqrt(2*self.DiffusionCoefficient*self.TimeSteps)
        for i in range(1,n):
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


    def generation_motion_rectangle(self, x_limit:float=1e-6):

        n = int(self.TotalTime/self.TimeSteps)
        x, y = np.zeros(n), np.zeros(n)
        old_x, old_y = self.StartingPoint
        x[0],y[0] = old_x,old_y
        factor = np.sqrt(2*self.DiffusionCoefficient*self.TimeSteps)
        for i in range(1,n):
            r1 =  np.random.normal(0,factor)
            r2 =  np.random.normal(0,factor)
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

class FRAPSimulator():

    def __init__(self, particle_density, time_steps, total_time, diffusion_coefficient, immobility):
        self.density = particle_density
        self.dt = time_steps
        self.t = total_time
        self.D = diffusion_coefficient
        self.mobility = 1 - immobility

    def generate_trajectories(self,interest_zone,frap_radius, confinement_size=np.inf,verbose=True):
        trajectories={}
        N = int(self.density*4*(interest_zone*1e6))
        n = int(self.t/self.dt)
        if verbose:
            print('Generating trajectories...')
        for i in range(N):
            center = np.random.uniform(-interest_zone,interest_zone,2)
            # Photobleach particles at the center of the region
            if (center[0]**2 + center[1]**2) >= frap_radius**2:
                fluorescence = True
                    # if particle moves
                if i < self.mobility*N:
                    Simulator = DiffusionSimulator(self.t,self.dt,self.D,center)
                    Simulator.generation_motion_circle(confinement_size)
                    trajectory = Simulator.data
                # if particle is immobile
                else:
                    trajectory = (center[0]*np.ones(n), center[1]*np.ones(n))
            #photobleached particles don't move
            else:
                fluorescence = False
                trajectory = (center[0]*np.ones(n), center[1]*np.ones(n))
            particle_data = {}
            particle_data['trajectory'] = trajectory    
            particle_data['fluorescence'] = fluorescence
            trajectories[f'{i}'] = particle_data
        self.trajectories = trajectories

    def generate_film(self,fov,pixel_size,filename='FRAP_simulator',saveframe=True,verbose=True):
        if verbose:
            print('Generating film...')
        x = np.mgrid[-fov:fov:pixel_size]
        images = []
        pixel_num = int(2*fov/pixel_size)
        def update(frame_id):
            image = np.zeros((pixel_num, pixel_num))
            for id in self.trajectories.keys():
                particle = self.trajectories[id]
                if particle['fluorescence'] or frame_id==0:
                    position = [particle['trajectory'][0][frame_id], particle['trajectory'][1][frame_id]]
                    if np.abs(position[0]) > fov or np.abs(position[1]) > fov:
                        continue
                    ix = find_nearest(x,position[0])
                    iy = find_nearest(x,position[1])
                    if ix >= pixel_num:
                        ix -= 1
                    if iy >= pixel_num:
                        iy -= 1
                    image[ix,iy] += 1
            images.append(image[1:-1,1:-1])
            a_image.set_data(image)

        fig,ax = plt.subplots()
        image = np.zeros((pixel_num, pixel_num))
        for id in self.trajectories.keys():
            particle = self.trajectories[id]
            position = [particle['trajectory'][0][0], particle['trajectory'][1][0]]
            if np.abs(position[0]) > fov or np.abs(position[1]) > fov:
                continue
            ix = find_nearest(x,position[0])
            iy = find_nearest(x,position[1])
            if ix >= pixel_num:
                ix -= 1
            if iy >= pixel_num:
                iy -= 1
            image[ix,iy] += 1
        a_image = ax.imshow(image,cmap='gray')
        ax.axis('off')
        ani=animation.FuncAnimation(fig=fig,func=update,frames=range(0,int(self.t/self.dt)))
        ani.save(f'{filename}.gif',writer='pillow')
        if saveframe:
            np.save(f'data\images_{filename}_{self.D}.npy', np.array(images))
        if verbose:
            print('Done!')
        return np.array(images)
    
    def generate_images(self,fov,pixel_size):
        x = np.mgrid[-fov:fov:pixel_size]
        images = []
        pixel_num = int(2*fov/pixel_size)
        N = int(self.t/self.dt)
        for frame_id in range(N):
            image = np.zeros((pixel_num, pixel_num))
            for id in self.trajectories.keys():
                particle = self.trajectories[id]
                if particle['fluorescence']:
                    position = [particle['trajectory'][0][frame_id], particle['trajectory'][1][frame_id]]
                    if np.abs(position[0]) > fov or np.abs(position[1]) > fov:
                        continue
                    ix = find_nearest(x,position[0])
                    iy = find_nearest(x,position[1])
                    if ix >= pixel_num:
                        ix -= 1
                    if iy >= pixel_num:
                        iy -= 1
                    image[ix,iy] += 1
            images.append(image[1:-1,1:-1])
        return np.array(images)
            


particles_density = 200 #particules par micron
dt = 0.005
time = 5
Ds = [2.5e-13, 1e-13, 0.5e-13,0.25e-13]
immobility = 0

sigma = 1e-6
radius = 0.5e-6
pixel_size = 0.1e-6
fov = 0.8*sigma
confinement = sigma

# Working many simulations


for D in Ds:
    image_set = []
    for _ in tqdm(range(10)):
        FRAP = FRAPSimulator(particles_density,dt,time,D,immobility)
        FRAP.generate_trajectories(sigma,radius,confinement,verbose=False)
        images = FRAP.generate_images(fov,pixel_size)
        image_set.append(images)
        plt.close()
    image_set = np.array(image_set)
    image_average = np.average(image_set, axis=0)
    np.save(f'average_set_{time}s_{dt}dt_{D}D.npy', image_average)


# Tracing FRAP curves, immobility

# data = [0,0.2,0.4,0.6,0.8]
# i = 0
# for info in data:
#     if info == 0:
#         filename = f'average_set_1.npy'
#         image_average = np.load(f'data\simulateur 2\{filename}')[2:]
#         radius_pixels = int(radius/pixel_size)
#         pix_num = image_average.shape[1]
#         ROI = np.zeros((pix_num, pix_num)) 
#         ROI[int(pix_num/2 - radius_pixels): int(pix_num/2+radius_pixels+1),int(pix_num/2 - radius_pixels): int(pix_num/2+radius_pixels+1)] = disk(radius_pixels)
#         t_array = np.linspace(0,3,image_average.shape[0])
#         image_average *= ROI
#         trace = np.sum(image_average,axis=(1,2))
#         max_value = np.average(trace[500:])
#         plt.plot(t_array,trace/max_value,color=colors[i],label = f'{round((1-info)*100)}%')
#         print(max_value)
#         i += 1
#     else:
#         filename = f'average_set_1_{info}.npy'
#         image_average = np.load(f'data\simulateur 2\{filename}')[2:]
#         radius_pixels = int(radius/pixel_size)
#         pix_num = image_average.shape[1]
#         ROI = np.zeros((pix_num, pix_num)) 
#         ROI[int(pix_num/2 - radius_pixels): int(pix_num/2+radius_pixels+1),int(pix_num/2 - radius_pixels): int(pix_num/2+radius_pixels+1)] = disk(radius_pixels)
#         t_array = np.linspace(0,3,image_average.shape[0])
#         image_average *= ROI
#         trace = np.sum(image_average,axis=(1,2))
#         plt.plot(t_array,trace/max_value,color=colors[i],label = f'{round((1-info)*100)}%')
#         i += 1
# plt.minorticks_on()
# plt.xlabel('Time (s)')
# plt.ylabel('Fluorescence normalisée (a.u.)')
# plt.legend()
# plt.savefig('FRAP_immobility.pdf')
# plt.show()

# FRAP curves, D

# data = [5.0,2.0,1.0,0.25,0.1]
i = 0
for info in Ds:
    filename = f'average_set_{time}s_{dt}dt_{info}D.npy'
    image_average = np.load(f'data\simulateur 2\{filename}')
    radius_pixels = int(radius/pixel_size)
    pix_num = image_average.shape[1]
    ROI = np.zeros((pix_num, pix_num)) 
    ROI[int(pix_num/2 - radius_pixels): int(pix_num/2+radius_pixels+1),int(pix_num/2 - radius_pixels): int(pix_num/2+radius_pixels+1)] = disk(radius_pixels)
    t_array = np.linspace(0,5,image_average.shape[0])
    image_average *= ROI
    trace = np.sum(image_average,axis=(1,2))
    if info == 2.5e-13:
        max_value = np.max(trace)
    plt.plot(t_array,trace/max_value,color=colors[i],label = f'D={round(info*1e12*100)/100}$\mu m^2/s$')
    i += 1
plt.minorticks_on()
plt.xlabel('Time (s)')
plt.ylabel('Fluorescence normalisée (a.u.)')
plt.legend()
plt.savefig('FRAP_D.pdf')
plt.show()


# # FRAP curves, confinement

# data = [5e-07,1e-06,'inf']
# labels = [f'Tube: 2R', f'Tube: 4R', 'Diffusion libre']
# i = 0
# for info in data:
#     filename = f'average_set_{info}.npy'
#     image_average = np.load(f'data\simulateur 2\{filename}')[2:]
#     radius_pixels = int(radius/pixel_size)
#     pix_num = image_average.shape[1]
#     ROI = np.zeros((pix_num, pix_num)) 
#     ROI[int(pix_num/2 - radius_pixels): int(pix_num/2+radius_pixels+1),int(pix_num/2 - radius_pixels): int(pix_num/2+radius_pixels+1)] = disk(radius_pixels)
#     t_array = np.linspace(0,5,image_average.shape[0])
#     image_average *= ROI
#     trace = np.sum(image_average,axis=(1,2))
#     if info == 5e-07:
#         max_value = np.average(trace)
#         print(max_value)
#     plt.plot(t_array,trace/max_value,color=colors[i],label = labels[i])
#     i += 1
# plt.minorticks_on()
# plt.xlabel('Time (s)')
# plt.ylabel('Fluorescence normalisée (a.u.)')
# plt.legend(loc=3)
# plt.savefig('FRAP_confinement.pdf')
# plt.show()



# Generate Gif

# image_average = np.load(r'average_set_0.5s_10e-5dt.npy')

# def update_image(frame_id):
#     image = image_average[frame_id]
#     a_image.set_data(image)

# fig,ax=plt.subplots()
# a_image = ax.imshow(image_average[0],cmap='gray')
# ax.axis('off')
# ani=animation.FuncAnimation(fig=fig,func=update_image,frames=tqdm(range(image_average.shape[0])),interval=3)
# ani.save(f'FRAP_{time}s_{dt}dt.gif',writer='pillow')
# # plt.show()