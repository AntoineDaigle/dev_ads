import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

colors = ['#003D5B', '#D1495B', '#EDAE49', '#00798C', '#401F3E','blue','green','black','red']

def find_peak_from_folder(path,total_time,time_peak):
    data = np.load(path)
    dt = total_time/data.shape[0]
    peak_value = np.average(data[int(time_peak/dt):])
    return peak_value



# Immobility

# immobilities = [0,0.2,0.4,0.6,0.8]
# time = 3 
# dt = 0.001
# peak_time = 1.5

# peaks = []

# for immobility in immobilities:
#     peak = find_peak_from_folder(f'data/simulateur 2/trace_{time}s_{dt}dt_{immobility}imm.npy',time, peak_time)
#     mobility = (1-immobility)*100
#     peaks.append((mobility,peak))

# peaks = np.array(peaks)

# plt.scatter(peaks[:,0], peaks[:,1]/np.max(peaks[:,1]),color = colors[0])
# plt.xlabel('Mobilité des partciules (%)')
# plt.ylabel('Fluorescence au plateau normalisée (a.u.)')
# plt.xlim(0)
# plt.ylim(0)
# plt.minorticks_on()
# plt.savefig(f'figures/simulateur 2/FRAP_recovery_immobility.pdf')
# plt.show()

# Recovery time

Ds = np.array([1e-13,2.2e-13,3.5e-13,5e-13, 1.5e-13,10e-13,30e-13,50e-13])

time = 3
dt=0.001
recovery_time = []   
for D in Ds:
    data = np.load(f'data/simulateur 2/trace_{time}s_{dt}dt_{D}D.npy')
    plateau = np.percentile(data,95)
    id = (np.abs(data - plateau/2)).argmin()
    recovery_time.append(id*dt)

def diff(d, a):
    return a/d

a, pcov = curve_fit(diff, Ds*1e12, recovery_time)
x = np.linspace(Ds[0]*1e12,Ds[-1]*1e12)
plt.scatter(Ds*1e12, recovery_time, color = colors[0])
plt.plot(x, diff(x,a), color = colors[3],alpha=0.6,ls='--')
plt.ylabel('Recovery half time (s)')
plt.xlabel(f'Coefficient de diffusion D ($\mu m^2/s$)')
plt.minorticks_on()
plt.savefig('recovery_time_diffusion.pdf')
plt.show()


# time = 3
# dt = 0.001
# time_array = np.linspace(0, time, int(3/dt))
# filename = f'{time}s_{dt}dt_{Ds[2]}D'
# data = np.load(f'data/simulateur 2/trace_{filename}.npy')
# plateau = np.percentile(data,95)
# id = (np.abs(data - plateau/2)).argmin()
# recovery_time = id*dt

# plt.plot(time_array, data, color = colors[0])
# plt.hlines(plateau/2,0,recovery_time,color = colors[3],alpha = 0.8,ls='--')
# plt.vlines(recovery_time,0,plateau/2,color=colors[3],alpha=0.8,ls='--')
# plt.xlabel('Time (s)')
# plt.xlim(0,0.8)
# plt.ylim(0)
# plt.ylabel('Fluorescence (a.u.)')
# plt.minorticks_on()
# plt.savefig(f'figures/simulateur 2/recovery_time_fluo_{filename}.pdf')
# plt.show()
