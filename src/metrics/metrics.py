import numpy as np
import cv2

# [TODO] Replace with actual upscaler
import tensorflow.keras.callbacks


def upscale_image(model, image):
    return image

def downscale(image, scale_factor):
    x,y, channels = image.shape
    return cv2.resize(image, (x//scale_factor, y//scale_factor,channels))

class ImageQualityCallback(tensorflow.keras.callbacks.Callback):
    def __init__(self, test_images, evaluate_function):
        super(ImageQualityCallback, self).__init__()
        self.test_images = test_images
        self.evaluate_function  = evaluate_function

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
                psnr_sum += self.evaluate_function(upscaled, downscaled_image)

            psnr_sum /= len(self.test_images)

            print("Average PSNR", psnr_sum)