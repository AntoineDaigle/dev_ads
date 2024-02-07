import numpy as np
import matplotlib.pyplot as plt
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

    def generation_motion_circle(self, radius = np.inf, obstacle=(0,0)):
        n = int(self.TotalTime/self.TimeSteps)
        x, y = np.zeros(n), np.zeros(n)
        old_x, old_y = self.StartingPoint
        x[0],y[0] = old_x,old_y
        obstacle_position, obstacle_length = obstacle
        factor = np.sqrt(2*self.DiffusionCoefficient*self.TimeSteps)
        colls = 0
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

            if obstacle_length != 0:
                if old_x <= obstacle_position and new_x >= obstacle_position and np.abs(old_y) < obstacle_length/2:
                    colls +=1
                    r = np.sqrt(r1**2 + r2**2)
                    angle = np.arctan(r2/r1)
                    temp_y = interp1d([old_x, new_x], [old_y, new_y],fill_value="extrapolate")(obstacle_position)
                    temp_x = obstacle_position
                    initial_to_wall = np.sqrt((temp_y - old_y)**2 + (temp_x - old_x)**2)

                    final_dist_remaining = r - initial_to_wall
                    new_angle = np.pi - angle

                    new_x = temp_x + final_dist_remaining*np.cos(new_angle)
                    new_y = temp_y + final_dist_remaining*np.sin(new_angle) 

                    if old_x <= obstacle_position and new_x >= obstacle_position:
                        new_x = temp_x
                        new_y = temp_y

                if old_x >= obstacle_position and new_x <= obstacle_position and np.abs(old_y) < obstacle_length/2:
                    colls +=1
                    r = np.sqrt(r1**2 + r2**2)
                    angle = np.arctan(r2/r1)
                    temp_y = interp1d([old_x, new_x], [old_y, new_y],fill_value="extrapolate")(obstacle_position)
                    temp_x = obstacle_position
                    initial_to_wall = np.sqrt((temp_y - old_y)**2 + (temp_x - old_x)**2)

                    final_dist_remaining = r - initial_to_wall
                    new_angle = 2*np.pi - angle

                    new_x = temp_x + final_dist_remaining*np.cos(new_angle)
                    new_y = temp_y + final_dist_remaining*np.sin(new_angle) 

                    if old_x >= obstacle_position and new_x <= obstacle_position:
                        new_x = old_x
                        new_y = old_x                      

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

    def generate_trajectories(self,interest_zone,frap_radius, confinement_size=np.inf,wall = (0,0),verbose=True):
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
                    Simulator.generation_motion_circle(confinement_size,obstacle=wall)
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
        

particles_densitys = [2] #particules par micron
dt = 1e-3
time = 2
# Ds = [0.1e-12, 0.5e-12, 1e-12, 5e-12]
D = 1e-12
immobility = 0

sigma = 3e-6
radius = 0
pixel_size = 0.15e-6
fov = sigma 
confinement = sigma
wall_pos = 0
wall = (0, 8*pixel_size)
# Working many simulations

# trajectory = DiffusionSimulator(time,dt,D,(0,0))
# trajectory.generation_motion_circle(confinement, wall)
# x,y = trajectory.data
# plt.plot(x,y)
# plt.vlines(wall[0],-wall[1]/2,wall[1]/2,color='k')
# plt.show()

for particles_density in particles_densitys:
    for i in tqdm(range(1)):
        image_set = []
        FRAP = FRAPSimulator(particles_density,dt,time,D,immobility)
        FRAP.generate_trajectories(sigma,radius,confinement,wall=wall,verbose=False)
        trajectories = FRAP.trajectories
        for particle in trajectories.keys():
            x,y  = trajectories[particle]['trajectory']
            plt.plot(x,y,label=particle)
        images = FRAP.generate_images(fov,pixel_size)
        image_set.append(images)
        image_set = np.array(image_set)
        # image_average = np.average(image_set, axis=0)
        fn = f'average_set_{time}s_{dt}dt_{D}D_{particles_density}em_{i}.npy'
        # np.save(f'data\simumulateur 4\mur/{fn}', image_set)

plt.vlines(wall[0], -wall[1]/2,wall[1]/2,color = 'k',linewidth=4)
# plt.axis('off')
plt.xlim(-sigma/2,sigma/2)
plt.ylim(-sigma/2, sigma/2)
plt.savefig(f'figures/simulateur 4/mur_partiel.pdf')
plt.show()


# Tracing FRAP curves, immobility

# data = [0,0.2,0.4,0.6,0.8]
# time = 3
# dt = 0.001
# i = 0
# for info in data:
#     filename = f'{time}s_{dt}dt_{info}imm'
#     image_average = np.load(f'data\simulateur 2\\average_set_{filename}.npy')
#     radius_pixels = int(radius/pixel_size)
#     pix_num = image_average.shape[1]
#     ROI = np.zeros((pix_num, pix_num)) 
#     ROI[int(pix_num/2 - radius_pixels): int(pix_num/2+radius_pixels+1),int(pix_num/2 - radius_pixels): int(pix_num/2+radius_pixels+1)] = disk(radius_pixels)
#     t_array = np.linspace(0,3,image_average.shape[0])
#     image_average *= ROI
#     trace = np.sum(image_average,axis=(1,2))
#     if info == 0:
#         max_value = np.average(trace[1000:-500])
#     np.save(f'data/simulateur 2/trace_{filename}.npy', trace)
#     plt.plot(t_array[:-500],trace[:-500]/max_value,color=colors[i],label = f'{round((1-info)*100)}%')
#     i += 1
# plt.minorticks_on()
# plt.xlabel('Time (s)')
# plt.ylabel('Fluorescence normalisée (a.u.)')
# plt.legend()
# plt.savefig('FRAP_immobility.pdf')
# plt.show()

# FRAP curves, D

# data = [1e-13,2.2e-13,2.5e-13,3.5e-13,5e-13]
# i = 0
# for info in Ds:
#     filename = f'{time}s_{dt}dt_{info}D'
#     image_average = np.load(f'data/simulateur 2/average_set_{filename}.npy')
#     radius_pixels = int(radius/pixel_size)
#     pix_num = image_average.shape[1]
#     ROI = np.zeros((pix_num, pix_num)) 
#     ROI[int(pix_num/2 - radius_pixels): int(pix_num/2+radius_pixels+1),int(pix_num/2 - radius_pixels): int(pix_num/2+radius_pixels+1)] = disk(radius_pixels)
#     t_array = np.linspace(0,time,image_average.shape[0])
#     image_average *= ROI
#     trace = np.sum(image_average,axis=(1,2))
#     if i==0:
#         max_value = np.max(trace)
#     np.save(f'data/simulateur 2/trace_{filename}.npy', trace)
#     plt.plot(t_array,trace/max_value,color=colors[i],label = f'D={round(info*1e12*100)/100}$\mu m^2/s$')
#     i += 1
# plt.minorticks_on()
# plt.xlabel('Time (s)')
# plt.ylabel('Fluorescence normalisée (a.u.)')
# plt.legend()
# # plt.savefig('FRAP_D.pdf')
# plt.show()


# # FRAP curves, confinement

# data = [5e-07,1e-06,'inf']
# labels = [f'Tube: 2R', f'Tube: 4R', 'Diffusion libre']
# i = 0
# for info in confinements:
#     filename = f'average_set_{time}s_{dt}dt_{info}.npy'
#     image_average = np.load(f'data\simulateur 2\{filename}')[2:]
#     radius_pixels = int(radius/pixel_size)
#     pix_num = image_average.shape[1]
#     ROI = np.zeros((pix_num, pix_num)) 
#     ROI[int(pix_num/2 - radius_pixels): int(pix_num/2+radius_pixels+1),int(pix_num/2 - radius_pixels): int(pix_num/2+radius_pixels+1)] = disk(radius_pixels)
#     t_array = np.linspace(0,5,image_average.shape[0])
#     image_average *= ROI
#     trace = np.sum(image_average,axis=(1,2))
#     if confinements[0]:
#         max_value = np.max(trace)
#     plt.plot(t_array,trace/max_value,color=colors[i],label = labels[i])
#     i += 1
# plt.minorticks_on()
# plt.xlabel('Time (s)')
# plt.ylabel('Fluorescence normalisée (a.u.)')
# plt.legend(loc=3)
# plt.savefig('FRAP_confinement.pdf')
# plt.show()


# Generate Gif
time = 10
dt = 0.001
pluses = ['']

# for plus in pluses:

#     fn = f'data/simulateur 4/5ems/average_set_10s_0.001dt_1e-12D_5em_0'

#     image_average = np.load(f'{fn}.npy')

#     def update_image(frame_id):
#         image = image_average[frame_id]
#         a_image.set_data(image)

#     fig,ax=plt.subplots()
#     a_image = ax.imshow(image_average[0],cmap='gray')
#     ax.axis('off')
#     ani=animation.FuncAnimation(fig=fig,func=update_image,frames=tqdm(range(image_average.shape[0])),interval=dt*1000/3)
#     ani.save(f'{fn}.gif',writer='pillow')
#     print('Done!')