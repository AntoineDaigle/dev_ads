import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import stats
from tqdm import tqdm

colors = ['#003D5B', '#D1495B', '#EDAE49', '#00798C', '#805E73','#475841','grey','black','red']

def extract_intensity(dataset, gaussian_beam_width=300e-9/2, pixel_size=0.15e-6,
                    position = (0,0), mask_validation=False):

    pixel_num = (dataset[0].shape)[0]
    # pixel_num = int(2 * FOV/ pixel_size)
    circle_center_x = int(pixel_num/2 + position[0])
    circle_center_y = int(pixel_num/2 + position[1])
    circle_radius = gaussian_beam_width/pixel_size

    mask = np.zeros(dataset[0].shape)
    mask = cv2.circle(mask, (circle_center_x, circle_center_y),
                      int(circle_radius), (255, 255, 255), -1)
    if mask_validation == True:
        plt.imshow(mask)
        plt.show()

    mask = np.uint8(mask)
    intensity = []

    for i in range(dataset.shape[0]):
        intensity.append(np.sum(cv2.bitwise_and(dataset[i,:,:],
                                                dataset[i,:,:],
                                                mask=mask)))

    return intensity

def corr_article(signal1, signal2)->np.array:
    """Function that calculate the autocorrelation by using the autocorrelation
    in the article.

    Args:
        signal (list): Signal to autocorrelate.

    Returns:
        np.array: The autocorrelation of the signal.
    """
    signal1 = np.asarray(signal1)
    signal2 = np.asarray(signal2)
    squared_mean = np.mean(signal1) * np.mean(signal2)

    corr = []
    for _ in range(signal1.shape[0]):
        corr.append((np.mean(signal1 * signal2) / squared_mean) - 1)
        signal1 = np.roll(signal1, 1)
    return corr

def autocorr_article(signal:list)->np.array:
    """Function that calculate the autocorrelation by using the autocorrelation
    in the article.

    Args:
        signal (list): Signal to autocorrelate.

    Returns:
        np.array: The autocorrelation of the signal.
    """
    array_sig = np.asarray(signal)
    squared_mean = np.mean(signal)**2

    signal_1, signal_2 = array_sig, array_sig

    autocorr = []
    for i in range(len(signal)):
        autocorr.append((np.mean(signal_1 * signal_2) / squared_mean) - 1)
        signal_1 = np.roll(signal_1, 1)

    return autocorr


fig, ax1 = plt.subplots(1,1, figsize=(10,5))
time = 10
dt = 1e-3
D = 1e-12
col=0

density = 15
densities = [5]
distance = 1

for density in densities:#, 1e-14, 1e-14, 3e-14, 6e-14]:
    correlations = []
    for i in range(15):
        if i in []:
            continue
        fn = f'average_set_{time}s_{dt}dt_{D}D_{density}em_{i}'
        dataset = np.load(f'data/simulateur 4/mur/{fn}.npy')[0]
        signal1 = extract_intensity(dataset, position=(distance,0),mask_validation=False)
        signal2 = extract_intensity(dataset, position=(-distance,0),mask_validation=False)
        correlation = corr_article(signal1, signal2)
        correlations.append(correlation)
    correlations = np.array(correlations)
    correlation = np.average(correlations,axis=0)
    time_stamp = np.arange(0, time, dt)
    ax1.plot(time_stamp, correlation, color= colors[2],label = 'Avec barrière')
    correlations = []
    for i in range(15):
        fn = f'average_set_{time}s_{dt}dt_{D}D_{density}em_{i}'
        dataset = np.load(f'data/simulateur 4/{density}ems/{fn}.npy')[0]
        signal1 = extract_intensity(dataset, position=(distance,0),mask_validation=False)
        signal2 = extract_intensity(dataset, position=(-distance,0),mask_validation=False)
        correlation = corr_article(signal1, signal2)
        correlations.append(correlation)

    correlations = np.array(correlations)
    correlation = np.average(correlations,axis=0)
    time_stamp = np.arange(0, time, dt)
    ax1.plot(time_stamp, correlation, color= colors[3], label = 'Sans barrière')
    correlations = []
    # for i in range(15):
    #     fn = f'average_set_{time}s_{dt}dt_{D}D_{density}em_{i}'
    #     dataset = np.load(f'data/simulateur 4/{density}ems/{fn}_shotnoise.npy')
    #     # for k in tqdm(range(dataset.shape[0])):
    #     #     for l in range(dataset.shape[1]):
    #     #         for m in range(dataset.shape[2]):
    #     #             dataset[k,l,m] = stats.poisson.rvs(dataset[k,l,m])
    #     # np.save(f'data/simulateur 4/{density}ems/{fn}_shotnoise.npy',dataset)
    #     signal1 = extract_intensity(dataset, position=(-distance,0),mask_validation=False)
    #     signal2 = extract_intensity(dataset, position=(distance,0),mask_validation=False)
    #     correlation = corr_article(signal1, signal2)
    #     # correlation = autocorr_article(signal1)
    #     correlations.append(correlation)
    # correlations = np.array(correlations)
    # correlation = np.average(correlations,axis=0)
    # time_stamp = np.arange(0, time, dt)
    # ax2.plot(time_stamp, correlation, color= colors[3])
    # col+=1

ax1.set_xlabel(r'Délai $\tau$ (s)')
ax1.set_ylabel(r'Cross correlation $G_{12}(\tau)$')

ax1.set_xscale("log")
ax1.set_xlim(dt,10)
# ax1.set_ylim(-0.3,0.5)
# ax2.set_xlabel(r'Délai $\tau$ (s)')
# ax2.set_ylabel(r'Cross correlation $G_{12}(\tau)$')
# ax2.set_xscale("log")
# ax2.set_xlim(dt,10)
# ax2.set_ylim(-0.3,0.5)
# ax1.set_title('Without shot noise')
ax1.legend()
# ax2.set_title('With shot noise')
plt.tight_layout()
plt.savefig(r'figures/simulateur 4/mur_partiel.pdf')
plt.show()
