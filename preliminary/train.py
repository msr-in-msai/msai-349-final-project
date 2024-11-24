import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import ast
from model import PointNet

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

def train_pointnet(model, train_loader, num_epochs=100, learning_rate=0.001, device='cuda'):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (points, labels) in enumerate(train_loader):
            # Move to device and ensure correct shape (B, C, N)
            points = points.to(device)
            labels = labels.to(device)
            points = points.permute(0, 2, 1)  # (B, N, 3) -> (B, 3, N)
            
            # Forward pass
            outputs = model(points)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {total_loss/len(train_loader):.4f}, '
              f'Accuracy: {100.*correct/total:.2f}%')
    
    return model


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset and dataloader
    dataset = PointCloudDataset('data_generation/output.csv')
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize model with correct number of classes
    num_classes = len(dataset.label_encoder.classes_)
    model = PointNet(num_classes=num_classes)
    
    # Train model
    trained_model = train_pointnet(
        model=model,
        train_loader=train_loader,
        num_epochs=100,
        learning_rate=0.001,
        device=device
    )
    
    # Save model
    torch.save(trained_model.state_dict(), 'pointnet_model.pth')
    print("Training completed and model saved!")

if __name__ == "__main__":
    main()