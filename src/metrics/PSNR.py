from functools import reduce

import numpy as np
import cv2

from tensorflow.keras.callbacks import Callback

def mse(image, generated_image):
    return ((image-generated_image)**2).mean()

def PSNR(image, generated_image):
    return 20*np.log10(np.max(image)) - 10*np.log10(mse(image, generated_image))


# [TODO] Replace with actual upscaler
def upscale_image(model, image):
    return image

def downscale(image, scale_factor):
    x,y, channels = image.shape
    return cv2.resize(image, (x//scale_factor, y//scale_factor,channels))

class PSNRCallback(Callback):
    def __init__(self, test_images):
        super(PSNRCallback, self).__init__()
        self.test_images = test_images

    # Store PSNR value in each epoch.
    def on_epoch_begin(self, epoch, logs=None):
        self.psnr = []

    def on_epoch_end(self, epoch, logs=None):
        print("Mean PSNR for epoch: %.2f" % (np.mean(self.psnr)))
        if epoch % 20 == 0:
            psnr_sum = 0

            for i, image in enumerate(self.test_images):
                downscaled_image = downscale(image, 2)
                upscaled = upscale_image(self.model, downscaled_image)
                psnr_sum += PSNR(upscaled, downscaled_image)

            psnr_sum /= len(self.test_images)

            print("Average PSNR", psnr_sum)


if __name__ == "__main__":
    fake = cv2.imread("fake.png")
    fake2 = cv2.imread("fake_low_resolution.png")
    real = cv2.imread("real.png")

    print(PSNR(fake, real))
    print(PSNR(fake2, real))