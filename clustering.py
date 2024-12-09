# clustering.py
import os
import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from feature_extraction import extract_features

image_folder = r'C:\Users\it\Desktop\logo generator\dataset\images\flickr_logos_27_dataset\flickr_logos_27_dataset_images'

def load_images(image_folder):
    image_paths = []
    images = []
    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(image_folder)
            img = cv2.imread(img_path)
            if img is not None:
                image_paths.append(img_path)
                images.append(cv2.resize(img, (224, 224)))
    return image_paths, np.array(images)

def extract_image_features(images):
    features = []
    for image in images:
        feature = extract_features(image)
        features.append(feature)
    return np.array(features)

def visualize_clusters(features, labels, n_clusters):
    plt.figure(figsize=(8, 6))
    for i in range(n_clusters):
        cluster_points = features[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {i}")
    plt.title("Clustering Visualization")
    plt.legend()
    plt.show()

def perform_clustering():
    image_paths, images = load_images(image_folder)
    features = extract_image_features(images)
    
    kmeans = KMeans(n_clusters=10, random_state=42)
    labels = kmeans.fit_predict(features)
    
    visualize_clusters(features, labels, 10)
    perform_clustering()
