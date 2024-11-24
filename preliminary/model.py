import torch
import torch.nn as nn

class PointNet(nn.Module):
    def __init__(self, num_classes=10):
        super(PointNet, self).__init__()
        self.activation = nn.ReLU()
        self.t_net3x3 = nn.Conv1d(3, 3, kernel_size=1)
        self.mlp1_1 = nn.Linear(3, 64)
        self.mlp1_2 = nn.Linear(64, 64)
        self.t_net64x64 = nn.Conv1d(64, 64, kernel_size=1)
        self.mlp2_1 = nn.Linear(64, 128)
        self.mlp2_2 = nn.Linear(128, 1024)
        self.mlp3_1 = nn.Linear(1024, 512)
        self.mlp3_2 = nn.Linear(512, 256)
        self.mlp3_3 = nn.Linear(256, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)

        # Tnet 3x3
        x = self.t_net3x3(x)
        x = self.activation(x)

        # MLP #1
        x = x.transpose(1, 2).contiguous()  # (batch_size, num_points, 3)
        x = x.view(-1, 3)  # (batch_size * num_points, 3)
        x = self.mlp1_1(x)
        x = self.activation(x)
        x = self.mlp1_2(x)
        x = self.activation(x)

        # Reshape for Tnet 64x64
        x = x.view(batch_size, num_points, 64)  # (batch_size, num_points, 64)
        x = x.transpose(1, 2).contiguous()  # (batch_size, 64, num_points)
        x = self.t_net64x64(x)
        x = self.activation(x)

        # MLP #2
        x = x.transpose(1, 2).contiguous()  # (batch_size, num_points, 64)
        x = x.view(-1, 64)  # (batch_size * num_points, 64)
        x = self.mlp2_1(x)
        x = self.activation(x)
        x = self.mlp2_2(x)
        x = self.activation(x)

        # Reshape for max pooling
        x = x.view(batch_size, num_points, 1024)  # (batch_size, num_points, 1024)

        # Max pooling across points
        x = torch.max(x, dim=1)[0]  # (batch_size, 1024)

        # MLP #3
        x = self.mlp3_1(x)
        x = self.activation(x)
        x = self.mlp3_2(x)
        x = self.activation(x)
        x = self.mlp3_3(x)
        x = self.softmax(x)

        return x