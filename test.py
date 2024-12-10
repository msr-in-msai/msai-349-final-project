import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import ast
from model import PointNet
from sklearn.metrics import accuracy_score, classification_report


class PointCloudDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        
        # Convert string representation of points to numpy arrays
        self.points = self.data['point'].apply(lambda x: np.array(ast.literal_eval(x.replace('];[', '], ['))))
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.data['label'])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        points = torch.FloatTensor(self.points[idx])
        label = torch.LongTensor([self.labels[idx]])[0]
        return points, label


def test_pointnet(model, test_loader, device='cuda'):
    model.eval()
    model = model.to(device)
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for points, labels in test_loader:
            # Move data to device
            points = points.to(device)
            labels = labels.to(device)
            points = points.permute(0, 2, 1)  # (B, N, 3) -> (B, 3, N)

            # Forward pass
            outputs, _, _ = model(points)

            # Predictions
            _, predicted = outputs.max(1)
            all_preds.append(predicted.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # Concatenate all predictions and labels
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return all_preds, all_labels


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load testing dataset
    print('Loading testing dataset...')
    test_dataset = PointCloudDataset('data_generation/test_isaac_sim_3d.csv')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print('Testing dataset loaded!')

    # Initialize model with the same number of classes
    num_classes = len(test_dataset.label_encoder.classes_)
    model = PointNet(num_classes=num_classes)
    print('Model initialized!')

    # Load saved model checkpoint
    checkpoint_path = 'pointnet_model.pth'
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Checkpoint loaded from {checkpoint_path}")

    # Test the model
    print("Testing the model...")
    predictions, labels = test_pointnet(model, test_loader, device)

    # Compute accuracy and print classification report
    accuracy = accuracy_score(labels, predictions)
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(labels, predictions, target_names=test_dataset.label_encoder.classes_))


if __name__ == "__main__":
    main()
