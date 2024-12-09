import cv2
import numpy as np

def extract_features(image):
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize to fixed dimensions
    resized_image = cv2.resize(gray_image, (128, 128))
    # Flatten the image into a 1D array (basic feature extraction)
    features = resized_image.flatten()
    return features
