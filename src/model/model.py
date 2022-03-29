import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def getModel(channels, upscale_factor):
    conv_args = {
        "activation": "relu",
        "kernel_initializer": "Orthogonal",
        "padding": "same",
    }
    model = keras.Sequential(
        [
            Conv2D(64, 5, **conv_args),
            Conv2D(64, 3, **conv_args),
            Conv2D(32, 3, **conv_args),
            Conv2D(channels * (upscale_factor ** 2), 3, **conv_args),
            tf.nn.depth_to_space(upscale_factor),
        ]
    )

    return model