import pandas as pd
import numpy as np
import ast
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Function to load and preprocess data
def load_data(csv_file):
    data = pd.read_csv(csv_file)
    
    # Convert points from string representation to numpy arrays
    points = data['point'].apply(lambda x: np.array(ast.literal_eval(x.replace('];[', '],[')))).tolist()
    points = np.array(points)  # Shape: (num_samples, 1024, 3)
    points = points.reshape(len(points), -1)  # Flatten to (num_samples, 1024*3)
    
    # Encode labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(data['label'])
    
    return points, labels, label_encoder

# Load training and testing data
train_points, train_labels, label_encoder = load_data("data_generation/train_isaac_sim_3d.csv")
test_points, test_labels, _ = load_data("data_generation/test_isaac_sim_3d.csv")

# Define KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)

# Train the classifier
print("Training KNN...")
knn.fit(train_points, train_labels)
print("Training complete.")

# Predict on the test set
print("Predicting on test set...")
predictions = knn.predict(test_points)
print("Prediction complete.")

# Evaluate the model
accuracy = accuracy_score(test_labels, predictions)
print(f"Accuracy: {accuracy:.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(test_labels, predictions, target_names=label_encoder.classes_))
