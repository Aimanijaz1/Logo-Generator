# image_processing.py
import cv2
import numpy as np

def preprocess_image(image):
    image = cv2.resize(image, (224, 224))  # Resize image to fit model input size
    return image

def normalize_image(image):
    image = image / 255.0  # Normalize the pixel values
    return image
