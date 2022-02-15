import cv2
import numpy as np

IMAGE_PATH = "katt.jpg"

# Loading image from disk with imread
image_array = cv2.imread(IMAGE_PATH)

print(f"Shape of image: {image_array.shape}")
print(f"Number of data points: {len(image_array.flatten())}")

# Displaying image until user inputs anything
cv2.imshow("Image", image_array)
cv2.waitKey(0)

# Resize image to half it's size
y, x, *_ = image_array.shape
resized_image = cv2.resize(image_array, (int(x / 2), int(y / 2)))

# Write image to disk
cv2.imwrite("ny_katt.jpg", resized_image)
