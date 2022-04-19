from tensorflow.keras.callbacks import EarlyStopping

def early_stopping():
    return EarlyStopping(
        monitor = "loss",
        min_delta = 0,
        patience = 4,
        verbose = 2,
        mode = "auto",
        baseline = None,
        restore_best_weights = False,
    )