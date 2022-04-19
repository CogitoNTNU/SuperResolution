import tensorflow as tf
import cv2
import numpy as np
# Use TF Ops to process.
def process_input(input, input_size, upscale_factor):
    input = tf.image.rgb_to_yuv(input)
    last_dimension_axis = len(input.shape) - 1
    y, u, v = tf.split(input, 3, axis=last_dimension_axis)
    return tf.image.resize(y, [input_size, input_size], method="area")

def process_input_cv2(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    last_dimension_axis = len(image.shape) - 1
    y, u, v = np.split(image, 3, axis=last_dimension_axis)
    return y

def process_target(input):
    input = tf.image.rgb_to_yuv(input)
    last_dimension_axis = len(input.shape) - 1
    y, u, v = tf.split(input, 3, axis=last_dimension_axis)
    return y

