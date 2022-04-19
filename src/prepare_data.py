import sys
from processing.convert_images import convert_images_in_folder
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_folder', type=str,
                        help='data folder')
    parser.add_argument('output_folder', type=str,)
    args = parser.parse_args()
    convert_images_in_folder(args.input_folder, args.output_folder)