import cv2
from skimage.metrics import structural_similarity
import numpy as np

from src.metrics.metrics import ImageQualityCallback


def SSIM(image, generated_image):
    return structural_similarity(image, generated_image, data_range=
            max(np.max(image), np.max(generated_image))-min(np.min(image), np.min(generated_image)), channel_axis=2)

def SSIMCallback(test_images):
    return ImageQualityCallback(test_images, SSIM)

# For testing
if __name__ == "__main__":
    fake = cv2.imread("fake.png")
    fake2 = cv2.imread("fake_low_resolution.png")
    real = cv2.imread("real.png")

    print(fake.shape)
    print(real.shape)

    print(SSIM(fake, real))
    print(SSIM(fake2, real))