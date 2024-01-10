import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.ndimage import gaussian_filter
##########################
### Rectangle confined ###
##########################


# x,y = np.load("data_cercle.npy")
# x = x[:int(x.size/20)]
# y = y[:int(y.size/20)]

# # def correct_position(data_array, limit=0):
# #     new_set = np.round(data_array*10**9 + limit/2*10**9)
# #     return new_set + abs(min(new_set))

# # x_values = correct_position(x, 1e-6).astype(int)
# # y_values = correct_position(y, 0).astype(int)
# # matrix = np.empty((max(y_values), max(x_values)))

# # matrix[y_values[0], x_values[0]] = 1


# limit=1e-6
# size=100


# x_grid, y_grid = np.meshgrid(
#     np.linspace(-limit, limit, size),
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
#        xlabel=r"Position [m]")#,
#     #    ylim=[-5*limit, 5*limit],
#     #    xlim=[-1.5* limit, 1.5*limit])
# plt.show()

# def update(frame):
#     data = gaussian(x[frame], y[frame], x_grid, y_grid, 208e-9)
#     background_noise = 0# np.random.normal(1, 5000, data.shape)
#     poisson_noise = 0 #np.random.poisson(10000, data.shape)
#     ax.pcolormesh(x_grid, y_grid, (data + poisson_noise + background_noise), cmap="gray")



# ani = animation.FuncAnimation(fig=fig, func=update, interval=10)
# ani.save("test2.gif", writer="pillow")
# plt.show()





#######################
### Circle confined ###
#######################

x,y = np.load("data_cercle.npy")
x = x[:int(x.size/20)]
y = y[:int(y.size/20)]

limit=1e-6
size=100


x_grid, y_grid = np.meshgrid(
    np.linspace(-limit, limit, size),
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
       xlabel=r"Position [m]")#,
    #    ylim=[-5*limit, 5*limit],
    #    xlim=[-1.5* limit, 1.5*limit])
plt.show()

def update(frame):
    data = gaussian(x[frame], y[frame], x_grid, y_grid, 208e-9)
    background_noise = 0# np.random.normal(1, 5000, data.shape)
    poisson_noise = 0 #np.random.poisson(10000, data.shape)
    ax.pcolormesh(x_grid, y_grid, (data + poisson_noise + background_noise), cmap="gray")



ani = animation.FuncAnimation(fig=fig, func=update, interval=10)
ani.save("test cre.gif", writer="pillow")
plt.show()