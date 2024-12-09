# Essential Libraries
import numpy as np  # For numerical computations
import pandas as pd  # For handling CSV files (labels)
import cv2  # For image processing and loading images

# Machine Learning Libraries
from sklearn.model_selection import train_test_split  # Splitting dataset into training and testing
from sklearn.svm import SVC  # Support Vector Machine classifier
from sklearn.metrics import classification_report, accuracy_score  # For model evaluation

# Model Saving and Loading
import joblib  # To save and load the trained model

# Feature Extraction (Assuming custom implementation)
from feature_extraction import extract_features  # Your custom feature extraction function


def train_logo_classifier(label_file):
    labels = pd.read_csv(label_file)
    X = []
    y = []

    for _, row in labels.iterrows():
        image = cv2.imread(row['image_path'])
        if image is None:
            print(f"Error reading image: {row['image_path']}")
            continue
        feature = extract_features(image)
        X.append(feature)
        y.append(row['label'])

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train SVM model
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)

    # Evaluate the model
    predictions = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Print classification report
    print(classification_report(y_test, predictions))

    # Save the model
    model_path = "logo_classifier.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved at {model_path}")

    return model, accuracy  # Return model and accuracy
