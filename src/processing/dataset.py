import string
import tensorflow as tf
from tensorflow import keras
import os

# Add error handling
# Choose training pictures

def get_train_and_validate_datasets(url: string, path: string, batch_size: int, dim: int):
    # url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"
    dir = keras.utils.get_file(origin=url, fname="BSR", utar=True)
    root = os.path.join(dir, path)
    train = keras.preprocessing.image_dataset_from_directory(
    path,
    batch_size=batch_size,
    image_size=(dim, dim),
    validation_split=0.2,
    subset="training",
    seed=5784,
    label_mode=None,
    )

    validate = keras.preprocessing.image_dataset_from_directory(
    path,
    batch_size=batch_size,
    image_size=(dim, dim),
    validation_split=0.2,
    subset="validation",
    seed=5784,
    label_mode=None,
    )

    return train, validate