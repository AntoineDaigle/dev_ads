import numpy as np
import matplotlib.pyplot as plt
import cv2


data = np.load("average_set_0.5s_10e-5dt.npy")


def extract_intensity(dataset, gaussian_beam_width=208e-9, pixel_size=0.05e-6,
                      sigma=0.5e-6, mask_validation=False):

    FOV = sigma/2

    pixel_num = (dataset[0].shape)[0]
    print(pixel_num)


    # pixel_num = int(2 * FOV/ pixel_size)
    circle_center_x = int(pixel_num/2)
    circle_center_y = int(pixel_num/2)
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



def autocorr_article(signal):
    array_sig = np.asarray(signal)
    squared_mean = np.square(np.mean(signal))

    signal_1 = np.concatenate((np.zeros(array_sig.size), array_sig), axis=None)
    signal_2 = np.concatenate((array_sig, np.zeros(array_sig.size)), axis=None)
    
    autocorr = []
    for i in range(len(signal)):
        autocorr.append((np.mean(signal_1 * signal_2) / squared_mean) - 1)
        signal_2 = np.roll(signal_2, 1)

    return autocorr



if __name__ == "__main__":

    fig, (ax1, ax2) = plt.subplots(2)
    for i in [1e-13, 1e-11]:#, 1e-14, 1e-14, 3e-14, 6e-14]:

        dataset = np.load(f"average_set_10s_0.001dt_{i}D_ANTOINE.npy")

        intensity = extract_intensity(dataset, mask_validation=False)
        # print(type(intensity))
        # # intensity -= np.mean(intensity)
        autocorr = autocorr_article(intensity)


        # autocorr = np.correlate(intensity, intensity, mode="full")
        # autocorr = autocorr[int(autocorr.size / 2):]
        # autocorr /= np.mean(intensity)**2
        # autocorr -= 1
        time_stamp = np.arange(0, 10, 0.001)

        ax1.plot(time_stamp, intensity)
        ax2.plot(time_stamp, autocorr, label=i)

    ax2.set_xscale("log")
    ax2.legend()
    plt.tight_layout()
    plt.show()