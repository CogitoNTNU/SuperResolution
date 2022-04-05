from model.example_model import get_model
from processing.generator import get_generators
from model.modelCheckpoint import get_checkpoint
from tensorflow.keras.models import load_model
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_folder', type=str,
                        help='data folder')
    parser.add_argument('-b', "--batch_size", type=int, default=32,)
    parser.add_argument('-u', "--upscale_factor", type=int, default=2,)
    parser.add_argument('-m', "--model_save_path", type=str, default="model/checkpoint",)
    parser.add_argument('-t', "--target_size", type=int, default=128,)
    parser.add_argument('-e', "--epochs", type=int, default=10,)
    parser.add_argument('-c', "--continue_training", action="store_true")
    args = parser.parse_args()

    checkpoint = get_checkpoint(args.model_save_path)
    target_size = args.target_size
    upscale_factor = args.upscale_factor
    batch_size = args.batch_size
    gen = get_generators(args.data_folder, (target_size//upscale_factor, target_size//upscale_factor), (target_size, target_size), batch_size)

    if args.continue_training:
        model = load_model(args.model_save_path)
    else:
        model = get_model(1, args.upscale_factor)
    model.compile(loss="mean_squared_error", optimizer="adam")
    model.fit_generator(gen, epochs=20, steps_per_epoch=int(4400/batch_size), callbacks=[checkpoint])