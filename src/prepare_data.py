import sys
from processing.convert_images import convert_images_in_folder
if __name__ == "__main__":
    _, input_folder, output_folder = sys.argv
    convert_images_in_folder(input_folder, output_folder)