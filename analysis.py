import numpy as np 
import matplotlib.pyplot as plt

colors = ['#003D5B', '#D1495B', '#EDAE49', '#00798C', '#401F3E']

msdx = np.load('data\simulateur 1\msd_xrec.npy')
msdy = np.load('data\simulateur 1\msd_ycir.npy')
msd = np.load('data\simulateur 1\msd_cir.npy')

#rec
dt = 0.005
t_plateau = 2
t_array = np.linspace(0,6,int(6/dt))


plateau = np.average(msdx[int(t_plateau/dt):])

plt.plot(t_array,msdx*1e6**2,color = colors[2],label='Simulation')
plt.hlines(plateau*1e6**2,0,6,color=colors[4],ls='--',alpha=0.9,label=f'$L^2/6$')
plt.minorticks_on()
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel(f'MSD [$\mu$m]')
# plt.savefig(r'figures\simulateur1\num4.3.pdf')
plt.show()
print(np.sqrt(12*plateau))

#cir
dt = 0.01
t_plateau = 6
t_array = np.linspace(0,20,int(20/dt))
keep = msd[int(t_plateau/dt):]
plateau=np.average(keep)

plt.plot(t_array,msd*1e6**2,color = colors[1],label='Simulation')
plt.hlines(plateau*(1e6**2),0,20,color=colors[4],ls='--',alpha=0.9,label=f'$a^2$')
plt.minorticks_on()
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel(f'MSD [$\mu$m]')
# plt.savefig(r'figures\simulateur1\num4.4.pdf')
plt.show()
print(np.sqrt(2*plateau))

t=20