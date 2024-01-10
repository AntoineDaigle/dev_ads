import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.ndimage import gaussian_filter


##########################
### Rectangle confined ###
##########################

x,y = np.load("animations\simulateur1\data_free.npy")
x = x[:int(4*x.size/20)]
y = y[:int(4*y.size/20)]

limit=1e-6
size=62


x_grid, y_grid = np.meshgrid(
    np.linspace(-4*limit, 4*limit, size),
    np.linspace(-4*limit, 4*limit, size))


def gaussian(x0, y0, x_grid, y_grid, sigma):
    return (1/(2*np.pi*sigma**2) * np.exp(-((x_grid-x0)**2/(2*sigma**2)
     + (y_grid-y0)**2/(2*sigma**2))))


fig, ax = plt.subplots()
ax.pcolormesh(x_grid, y_grid, gaussian(x[0], y[0], 
                                       x_grid, y_grid, 208e-9), cmap="gray")
ax.minorticks_on()
ax.axis("equal")
ax.set(ylabel=r"Position [m]",
       xlabel=r"Position [m]")
plt.show()

def update(frame):
    data = gaussian(x[frame], y[frame], x_grid, y_grid, 208e-9)
    background_noise = 0# np.random.normal(1, 5000, data.shape)
    poisson_noise = 0 #np.random.poisson(10000, data.shape)
    ax.pcolormesh(x_grid, y_grid, (data + poisson_noise + background_noise), cmap="gray")



ani = animation.FuncAnimation(fig=fig, func=update, interval=10)
ani.save("animation free.gif", writer="pillow")
plt.show()




##########################
### Rectangle confined ###
##########################

# x,y = np.load("data_rectangle.npy")
# x = x[:int(x.size/20)]
# y = y[:int(y.size/20)]

# limit=1e-6
# size=62


# x_grid, y_grid = np.meshgrid(
#     np.linspace(-4*limit, 4*limit, size),
#     np.linspace(-4*limit, 4*limit, size))


# def gaussian(x0, y0, x_grid, y_grid, sigma):
#     return (1/(2*np.pi*sigma**2) * np.exp(-((x_grid-x0)**2/(2*sigma**2)
#      + (y_grid-y0)**2/(2*sigma**2))))


# fig, ax = plt.subplots()
# ax.pcolormesh(x_grid, y_grid, gaussian(x[0], y[0], 
#                                        x_grid, y_grid, 208e-9), cmap="gray")
# ax.minorticks_on()
# ax.axvline(limit/2)
# ax.axvline(limit/-2)
# ax.axis("equal")
# ax.set(ylabel=r"Position [m]",
#        xlabel=r"Position [m]")
# plt.show()

# def update(frame):
#     data = gaussian(x[frame], y[frame], x_grid, y_grid, 208e-9)
#     background_noise = 0# np.random.normal(1, 5000, data.shape)
#     poisson_noise = 0 #np.random.poisson(10000, data.shape)
#     ax.pcolormesh(x_grid, y_grid, (data + poisson_noise + background_noise), cmap="gray")



# ani = animation.FuncAnimation(fig=fig, func=update, interval=10)
# ani.save("animation tube.gif", writer="pillow")
# plt.show()





#######################
### Circle confined ###
#######################

# x,y = np.load("data_cercle.npy")
# x = x[int(16 * x.size/20):int(18 * x.size/20)]
# y = y[int(16 * y.size/20):int(18 * y.size/20)]

# limit=2e-6
# size=46 # 46 pixels alors 130 nm par pixel


# x_grid, y_grid = np.meshgrid(
#     np.linspace(-1.5*limit, 1.5*limit, size),
#     np.linspace(-1.5*limit, 1.5*limit, size))

# def gaussian(x0, y0, x_grid, y_grid, sigma):
#     return (1/(2*np.pi*sigma**2) * np.exp(-((x_grid-x0)**2/(2*sigma**2)
#      + (y_grid-y0)**2/(2*sigma**2))))


# fig, ax = plt.subplots()
# ax.pcolormesh(x_grid, y_grid, gaussian(x[0], y[0],
#                                        x_grid, y_grid, 208e-9), cmap="gray")
# ax.minorticks_on()
# ax.axis("equal")
# ax.set(ylabel=r"Position [m]",
#        xlabel=r"Position [m]")
# circle = plt.Circle((0,0), 2e-6, color='red', fill=False, linewidth=2)
# ax.add_patch(circle)
# plt.show()

# def update(frame):
#     data = gaussian(x[frame], y[frame], x_grid, y_grid, 208e-9)
#     background_noise = np.random.normal(1, 5000, data.shape)
#     poisson_noise = np.random.poisson(10000, data.shape)
#     ax.pcolormesh(x_grid, y_grid, (data + poisson_noise + background_noise), cmap="gray")
#     ax.add_patch(circle)



# ani = animation.FuncAnimation(fig=fig, func=update, interval=10)
# ani.save("animation cercle.gif", writer="pillow")
# plt.show()