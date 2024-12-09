# labeling_utils.py
import pandas as pd

def load_labels(label_file):
    return pd.read_csv(label_file)

def display_sample_labels(label_file, n=5):
    labels = load_labels(label_file)
    print(labels.head(n))

def filter_images_by_label(label_file, label):
    labels = load_labels(label_file)
    filtered_images = labels[labels['label'] == label]
    return filtered_images
