from functools import reduce

import numpy as np
import cv2

from tensorflow.keras.callbacks import Callback

from src.metrics.metrics import ImageQualityCallback


def mse(image, generated_image):
    return ((image-generated_image)**2).mean()

def PSNR(image, generated_image):
    return 20*np.log10(np.max(image)) - 10*np.log10(mse(image, generated_image))

def PSNRCallback(test_images):
    return ImageQualityCallback(test_images, PSNR)

# For testing
if __name__ == "__main__":
    fake = cv2.imread("fake.png")
    fake2 = cv2.imread("fake_low_resolution.png")
    real = cv2.imread("real.png")

    print(PSNR(fake, real))
    print(PSNR(fake2, real))