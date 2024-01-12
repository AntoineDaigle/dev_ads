import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import matplotlib.animation as animation
from tqdm import tqdm

colors = ['#003D5B', '#D1495B', '#EDAE49', '#00798C', '#401F3E']

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

particles_density = 20000 #particules par micron
dt = 0.01
time = 20
D = 2.5e-13
immobility = 0


mobility = 1-immobility
sigma = 1e-6
radius = 1e-7
trajectories = {}
N = int(particles_density*2*(sigma*1e6))

pixel_size = 0.5e-7
fov = 0.9*sigma
pixel_num = int(2*fov/pixel_size)


#generate random trajectories
print('Generating trajectories...')
for i in tqdm(range(N)):
    center = np.random.uniform(-sigma,sigma,2)   
    
    # Photobleach particles at the center of the region
    if (center[0]**2 + center[1]**2) >= radius**2:
        fluorescence = True
            # if particle moves
        if i < mobility*particles_density:
            Simulator = DiffusionSimulator(time,dt,D,center)
            Simulator.generation_motion_rectangle(2e-6)
            trajectory = Simulator.data
        # if particle is immobile
        else:
            trajectory = (center[0]*np.ones(int(time/dt)), center[1]*np.ones(int(time/dt)))
    
    #photobleached particles don't move
    else:
        fluorescence = False
        trajectory = (center[0]*np.ones(int(time/dt)), center[1]*np.ones(int(time/dt)))

    particle_data = {}
    particle_data['trajectory'] = trajectory    
    particle_data['fluorescence'] = fluorescence
    trajectories[f'{i}'] = particle_data


print('Generating film...')
x = np.mgrid[-fov:fov:pixel_size]
y = np.mgrid[-fov:fov:pixel_size]
images = []
def update(frame_id):
    image = np.zeros((pixel_num, pixel_num))
    for id in trajectories.keys():
        particle = trajectories[id]
        if particle['fluorescence']:
            position = [particle['trajectory'][0][frame_id], particle['trajectory'][1][frame_id]]
            if np.abs(position[0]) > fov or np.abs(position[1]) > fov:
                continue
            ix = find_nearest(x,position[0])
            iy = find_nearest(x,position[1])
            image[ix,iy] += 1
    images.append(image)
    a_image.set_data(image)


fig,ax = plt.subplots()
image = np.zeros((pixel_num, pixel_num))
for id in trajectories.keys():
    particle = trajectories[id]
    if particle['fluorescence']:
        position = [particle['trajectory'][0][90], particle['trajectory'][1][90]]
        if np.abs(position[0]) > fov or np.abs(position[1]) > fov:
            continue
        ix = find_nearest(x,position[0])
        iy = find_nearest(x,position[1])
        image[ix,iy] += 1
    images.append(image)

a_image = ax.imshow(image,cmap='gray')
ax.axis('off')
ani=animation.FuncAnimation(fig=fig,func=update,frames=range(0,int(time/dt)),interval=int(dt*1000))
ani.save('FRAP_simulator.gif',writer='pillow')
np.save('data\images_FRAP.npy', np.array(images))

print('Done!')