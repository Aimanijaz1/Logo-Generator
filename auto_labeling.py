import os
import numpy as np
import cv2
from sklearn.cluster import KMeans
import pandas as pd
from feature_extraction import extract_features  # Ensure this is implemented

# Paths
image_folder = r'C:\Users\it\Desktop\logo generator\dataset\images\flickr_logos_27_dataset\flickr_logos_27_dataset_images'
label_file = r'C:\Users\it\Desktop\logo generator\dataset\labels.csv'

# Function to load and resize images
def load_and_resize_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at {image_path} could not be loaded.")
    resized_image = cv2.resize(image, (224, 224))
    return resized_image

def load_images(image_folder):
    image_paths = []
    images = []
    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(image_folder, filename)
            try:
                img = load_and_resize_image(img_path)
                images.append(img)
                image_paths.append(img_path)
            except ValueError as e:
                print(e)  # Print error and continue with the next image
    return image_paths, np.array(images)

# Function to extract features from images
def extract_image_features(images):
    features = []
    for i, image in enumerate(images):
        try:
            feature = extract_features(image)
            features.append(feature)
        except Exception as e:
            print(f"Error extracting features for image {i}: {e}")
    return np.array(features)

def cluster_images(features, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(features)
    return kmeans.labels_

def save_labels(image_paths, labels, label_file):
    df = pd.DataFrame({'image_path': image_paths, 'label': labels})
    try:
        df.to_csv(label_file, index=False)
        print(f"Labels saved to {label_file}")
    except Exception as e:
        print(f"Error saving labels: {e}")
def main():
    print("Loading images...")
    image_paths, images = load_images(image_folder)
    if len(images) == 0:
        print("No images found. Please check the image folder path.")
        return

    print("Extracting features...")
    features = extract_image_features(images)
    if len(features) == 0:
        print("No features extracted.")
        return

    print("Clustering images...")
    labels = cluster_images(features)

    print("Saving labels...")
    save_labels(image_paths, labels, label_file)

main()
