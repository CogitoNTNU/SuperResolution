from model.example_model import get_model
from processing.generator import get_generators

BATCH_SIZE = 32

if __name__ == "__main__":
    gen = get_generators("../data2", (64, 64), (128, 128), BATCH_SIZE)
    model = get_model(1, 2)
    model.compile(loss="mean_squared_error", optimizer="adam")
    print(model.summary())
    model.fit_generator(gen, epochs=1, steps_per_epoch=1000)
