import tensorflow as tf

def get_checkpoint(filename):
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=filename,
        monitor='loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='low',
        save_freq='epoch',
        options=None,
        initial_value_treshold=None
    )