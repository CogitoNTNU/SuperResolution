import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D

def get_model(channels=3, upscale_factor=1):
    conv_args = {
        "activation": "relu",
        "kernel_initializer": "Orthogonal",
        "padding": "same",
    }


    inputs = keras.Input(shape=(64, 64, channels))
    x = Conv2D(64, 5, **conv_args)(inputs)
    x = Conv2D(64, 3, **conv_args)(x)
    x = Conv2D(32, 3, **conv_args)(x)
    x = Conv2D(channels * (upscale_factor ** 2), 3, **conv_args)(x)
    outputs = tf.nn.depth_to_space(x, upscale_factor)

#TODO add other layers than only conv2d
    return keras.Model(inputs, outputs)