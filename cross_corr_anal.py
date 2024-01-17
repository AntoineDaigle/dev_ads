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



if __name__ == "__main__":
    dataset = np.load("average_set_0.5s_10e-5dt.npy")

    intensity = extract_intensity(dataset, mask_validation=False)

    autocorr = np.correlate(intensity, intensity, mode="full")
    autocorr = autocorr[int(autocorr.size / 2):]

    time_stamp = np.arange(0, 0.5, 1e-5)

    fig, (ax1, ax2) = plt.subplots(2)

    ax1.plot(time_stamp[:-1], intensity)

    ax2.plot(time_stamp[:-1], autocorr)
    ax2.set_xscale("log")
    plt.tight_layout()
    plt.show()