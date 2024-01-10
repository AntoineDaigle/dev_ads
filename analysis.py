import numpy as np 
import matplotlib.pyplot as plt

msdx = np.load('msd_xrec.npy')
msdy = np.load('msd_ycir.npy')
msd = np.load('msd_cir.npy')
dt = 0.0005
t_plateau = 2


plateau = np.average(msdx[int(t_plateau*dt):])
print(np.sqrt(6*plateau))