
import cv2


def scale_image(img, scale):
    width = int(img.shape[1]*scale)
    height = int(img.shape[0]*scale)
    dim = [width, height]
    resized = cv2.resize(img, dim)
    return resized


def normalize_img(img):
    norm_dataset = img/255
    return norm_dataset


