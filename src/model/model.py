import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, MaxPool2D

def getModel(channels=3, upscale_factor=1):
    conv_args = {
        "activation": "relu",
        "kernel_initializer": "Orthogonal",
        "padding": "same",
    }
    model = keras.Sequential(
        [
            Input(shape=(None,None, channels)),
            Conv2D(64, 5, **conv_args),
            Conv2D(64, 3, **conv_args),
            Conv2D(channels * (upscale_factor ** 2), 3, **conv_args),
            MaxPool2D(),
            Conv2D(32, 3, **conv_args),
            Conv2D(channels * (upscale_factor ** 2), 3, **conv_args),
            tf.nn.depth_to_space(upscale_factor),
        ]
    )
#TODO add other layers than only conv2d
    return model