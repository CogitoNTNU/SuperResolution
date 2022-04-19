from tensorflow.keras.models import load_model
import cv2
from processing.generator import get_generators
import argparse
import os
import tensorflow as tf

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

    model = load_model(args.model_save_path, custom_objects={'tf': tf})

    target_size = args.target_size
    upscale_factor = args.upscale_factor
    batch_size = args.batch_size
    gen = get_generators(args.data_folder, (target_size//upscale_factor, target_size//upscale_factor), (target_size, target_size), batch_size)

    images = next(gen)

    predictions = model.predict(images[0])

    os.makedirs("results", exist_ok=True)
    print(images[0][0].shape)
    for i, image in enumerate(images[0]):
        cv2.imwrite(f"results/{i}_before.jpg", images[0][i]*255)
        cv2.waitKey()
        cv2.imwrite(f"results/{i}_after.jpg", predictions[i]*255)
        cv2.waitKey()
        cv2.imwrite(f"results/{i}_naive.jpg", cv2.resize(images[0][i], (target_size, target_size))*255)
        cv2.waitKey()