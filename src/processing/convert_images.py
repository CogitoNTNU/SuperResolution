import os
import cv2

from processing.dataprocessing import process_input_cv2

HR_FOLDER_NAME = "HR"

def convert_images_in_folder(input_folder, output_folder):
    paths = os.listdir(input_folder)

    os.makedirs(output_folder, exist_ok=True)
    hr_folder_path = os.path.join(output_folder, HR_FOLDER_NAME)
    os.makedirs(hr_folder_path, exist_ok=True)
    length = len(paths)
    for i, path in enumerate(paths):
        try:
            filename_hr = os.path.join(hr_folder_path, path)

            image = cv2.imread(os.path.join(input_folder, path))

            output_image = process_input_cv2(image)

            cv2.imwrite(filename_hr, output_image)
            print(f"{i} of {length}")
        except Exception as e:
            print(e)



if __name__ == "__main__":
    convert_images_in_folder("./../../flowers", "./../../data2")