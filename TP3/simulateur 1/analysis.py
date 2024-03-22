import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit

colors = ['#005C69', '#023618', '#E9B44C', '#B0413E', '#83C5BE']

def SpatialAutocorrelation(image):
    """Fonction pour que antoine comprenne mon code  (tu le comprendra pas mon esti)
    There is still a factor 10 in the results (about 10)

    Args:
        image (array): image to do the autocorrelation of

    Returns:
        autocorr (array): array of image with the autocorrelation function following equation 7 from the paper
    """
    shape = image.shape # Calculating the shape of the image
    image = image.ravel() # Flattening the image so it is a 1D array
    mean_image = np.average(image) # Calculating the mean pixel intensity
    normalized_image = image - mean_image # Normalizing following equation 2
    signal_1, signal_2 = normalized_image, normalized_image # assigning the normalized image to both signal 1 and signal 2 
    autocorr = [] #creating a list to put our autocorrelation values in
    for _ in range(image.shape[0]): #iterating for the shape of the array
        autocorr.append(np.average(signal_1*signal_2)/mean_image**2) # Equation 7
        signal_1 = np.roll(signal_1, 1) # Moving signal_1
    autocorr = np.array(autocorr) # transforming autocorrelation in array for futur operations
    return  np.fft.ifftshift(autocorr.reshape(shape)) # reshaping into image shape and shifting to bring the peaks at the center

def AutocorrelationFit(data,a,b,c):
    x,y = np.meshgrid(data[0],data[1])
    center=x.shape[1]/2
    # center = 10
    pos = np.dstack((x,y))
    gaussienne = stats.multivariate_normal([center, center], [[b**2,0],[0,b**2]]).pdf(pos)
    gaussienne /= np.max(gaussienne)
    return (a*gaussienne + c).ravel()

def density_analysis():
    densities = [0.1,0.2,.5,0.8,1,2,5,10]
    dp = []

    for density in densities:
        fit = np.load(f'data/density/fit_1e-07s_1e-12_2e-07_1e-07_{density}ems_500photons.npy')
        amps = 1/fit[:,1]
        psfs = fit[:,2]*100e-9
        psf_sem = stats.sem(psfs)
        psf_ave = np.mean(psfs)
        amp_sem = stats.sem(amps)
        amp_ave = np.mean(amps)
        dp.append([amp_ave, amp_sem,psf_ave,psf_sem])
    dp = np.array(dp)

    plt.plot([0,10],[0,10],ls='--',  color = colors[0], zorder = 1)
    plt.errorbar(densities, dp[:,0], xerr = 0, yerr = amp_sem, color = colors[1],zorder = 2,capsize=2, fmt = '.')
    plt.xlabel('Predicted density (ems/psf)')
    plt.ylabel('Real density (ems/psf)')
    plt.show()

    # plt.errorbar(densities, dp[:,2]*1e9, xerr=0,yerr=dp[:,3]*1e9, color = colors[1],zorder=1,capsize=2,fmt='.')
    # plt.plot([0,10],[200,200],ls='--', color = colors[0],zorder = 0)
    # plt.xlabel('Densité (éms/faisceau)')
    # plt.ylabel('PSF prédite du faisceau (nm)')
    # plt.show()

def intensity_and_noise_analysis():
    intensities = [10,100,200,500,1000]
    dp_noiseless = []
    dp_noisy = []
    for intensity in intensities:
        fit = np.load(f'data/intensity/fit_1e-07s_1e-12_2e-07_1e-07_0.2ems_{intensity}photons.npy')
        amps = 1/fit[:,1]
        psfs = fit[:,2]*100e-9
        psf_sem = stats.sem(psfs)
        psf_ave = np.mean(psfs)
        amp_sem = stats.sem(amps)
        amp_ave = np.mean(amps)
        dp_noiseless.append([amp_ave, amp_sem,psf_ave,psf_sem])
        fit = np.load(f'data/noise/fit_1e-07s_1e-12_2e-07_1e-07_0.2ems_{intensity}photons.npy')
        amps = 1/fit[:,1]
        psfs = fit[:,2]*100e-9
        psf_sem = stats.sem(psfs)
        psf_ave = np.mean(psfs)
        amp_sem = stats.sem(amps)
        amp_ave = np.mean(amps)
        dp_noisy.append([amp_ave, amp_sem,psf_ave,psf_sem])
    dp_noiseless = np.array(dp_noiseless)
    dp_noisy = np.array(dp_noisy)

    # plt.errorbar(intensities,dp_noiseless[:,0],xerr = 0, yerr = dp_noiseless[:,1], color = colors[1],zorder = 2,capsize=2, fmt = '.')
    # plt.errorbar(intensities,dp_noisy[:,0],xerr=0,yerr=dp_noisy[:,1], color = colors[0],zorder=2, capsize=2,fmt='.')
    # plt.hlines(0.2,0,1000,color = colors[0],ls='--',zorder=0)
    # plt.ylabel('Predicted density (ems/psf)')
    # plt.xlabel('Particle intensity (photons)')
    # plt.show()
    plt.errorbar(intensities,dp_noiseless[:,2]*1e9,xerr=0,yerr=dp_noiseless[:,3]*1e9,color=colors[1],zorder=2,fmt='.',capsize=2)
    plt.errorbar(intensities,dp_noisy[:,2]*1e9,xerr=0,yerr=dp_noisy[:,3]*1e9,fmt='.',color=colors[3],capsize=2,zorder=2)
    plt.xlabel('Particle intensity (ems/psf)')
    plt.ylabel('Predicted PSF (nm)')
    plt.show()
 
def duration_analysis():
    durations = [1e-7,5e-7,1e-6,5e-6,1e-5,5e-5]
    dp = []
    for duration in durations:
        fit = np.load(f'data/duration/fit_{duration}s_1e-12_2e-07_1e-07_0.2ems_500photons.npy')
        amps = 1/fit[:,1]
        psfs = fit[:,2]*100e-9
        psf_sem = stats.sem(psfs)
        psf_ave = np.mean(psfs)
        amp_sem = stats.sem(amps)
        amp_ave = np.mean(amps)
        dp.append([amp_ave, amp_sem,psf_ave,psf_sem])
    dp = np.array(dp)
    plt.plot([1e-7,1e-5],[0,10],ls='--',  color = colors[0], zorder = 1)
    plt.errorbar(durations, dp[:,0], xerr = 0, yerr = amp_sem, color = colors[1],zorder = 2,capsize=2, fmt = '.')
    plt.xlabel('Predicted density (ems/psf)')
    plt.ylabel('Real density (ems/psf)')
    plt.show()


# autocorr = SpatialAutocorrelation(image)
# x, y = np.arange(image.shape[0]), np.arange(image.shape[0])
# popt, pcov = curve_fit(AutocorrelationFit, (x,y), autocorr.ravel())
# amplitude, std, baseline = popt
# # plt.plot(autocorr.ravel(),color = colors[0], label = 'Autocorrelation example')
# # plt.plot(AutocorrelationFit((x,y),amplitude, std, baseline), color = colors[4], alpha=0.7, label = 'Fit')
# # plt.xlabel('Pixel number')
# # plt.ylabel(r'G($\xi$,$\nu$)')
# # plt.legend()
# # plt.savefig(f'simulateur 1/figures/fit.pdf')
# # plt.show()