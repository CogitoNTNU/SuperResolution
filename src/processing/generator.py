from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2

def get_generator(folder, batch_size=32, target_size=(128,128)):
    generator = ImageDataGenerator(rescale=1/255)
    return generator.flow_from_directory(folder, batch_size=batch_size, target_size=target_size, class_mode=None, color_mode="grayscale", seed=101)

def get_generators(folder, input_size, output_size, batch_size=32):
    return get_generator(folder, batch_size, input_size), get_generator(folder, batch_size, output_size)
if __name__ == "__main__":
    gen, gen2 = get_generators("../../flowers", (16,16), (128,128), 32)
    images = next(gen)
    print(images[0].shape)
    cv2.imshow("e", images[0])
    cv2.waitKey()

    images = next(gen2)
    print(images[0].shape)
    cv2.imshow("e", images[0])
    cv2.waitKey()