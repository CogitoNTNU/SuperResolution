import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, MaxPool2D, Lambda

def getModel(channels=1, upscale_factor=2):
    conv_args = {
        "activation": "relu",
        "kernel_initializer": "Orthogonal",
        "padding": "same",
    }
    model = keras.Sequential(
        [
            Input(shape=(None,None, channels)),
            Conv2D(64, 5, **conv_args),
            Conv2D(channels * (upscale_factor ** 2), 3, **conv_args),
            Conv2D(64, 3, **conv_args),
            MaxPool2D(),
            Conv2D(32, 3, **conv_args),
            Conv2D(channels * (upscale_factor ** 2), 3, **conv_args),
            Lambda(lambda x: tf.nn.depth_to_space(x, upscale_factor)),
        ]
    )
    return model