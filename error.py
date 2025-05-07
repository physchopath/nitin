import cv2
import numpy as np

# Load an image (ensure the path is correct)
image = cv2.imread('model/ck+/anger/S010_004_00000017.png')

# Check the shape and dtype of the loaded image
print("Shape of the image:", image.shape)
print("Data type of the image:", image.dtype)

# Check if the image contains complex numbers
if np.iscomplexobj(image):
    print("Image contains complex numbers.")
else:
    print("Image does not contain complex numbers.")
