import joblib
import cv2
from feature_extraction import extract_features

def predict_logo_type(model_path, image_path):
    model = joblib.load(model_path)  # Load trained model
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at {image_path} could not be loaded.")
    
    # Preprocess and extract features
    feature = extract_features(image)
    # Predict
    label = model.predict([feature])
    return label
