from model.example_model import get_model
from processing.generator import get_generators
from model.modelCheckpoint import get_checkpoint
from tensorflow.keras.models import load_model
BATCH_SIZE = 128
MODEL_SAVE_PATH = "model/checkpoint"
CONTINUE_TRAINING = False
UPSCALE_FACTOR = 2
TARGET_SIZE = 128
if __name__ == "__main__":
    checkpoint = get_checkpoint(MODEL_SAVE_PATH)
    gen = get_generators("../data2", (TARGET_SIZE//UPSCALE_FACTOR, TARGET_SIZE//UPSCALE_FACTOR), (TARGET_SIZE, TARGET_SIZE), BATCH_SIZE)
    if CONTINUE_TRAINING:
        model = load_model(MODEL_SAVE_PATH)
    else:
        model = get_model(1, UPSCALE_FACTOR)
    model.compile(loss="mean_squared_error", optimizer="adam")
    print(model.summary())
    model.fit_generator(gen, epochs=10, steps_per_epoch=int(4400/BATCH_SIZE), callbacks=[checkpoint])
