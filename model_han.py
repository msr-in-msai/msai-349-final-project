import torch
import torch.nn as nn


class Tnet(nn.Module):
    """
    It's composed of a shared MLP(64, 128, 1024)
    network (with layer output sizes 64, 128, 1024) on each
    point, a max pooling across points and two fully connected
    layers with output sizes 512, 256. The output matrix is
    initialized as an identity matrix. All layers, except the last
    one, include ReLU and batch normalization.

    The second transformation network has the same architecture
    as the first one except that the output is a 64X64 matrix.
    The matrix is also initialized as an identity. A
    regularization loss (with weight 0.001) is added to the
    softmax classification loss to make the matrix close to
    orthogonal.
    """

    def __init__(self,
                 k: int = 3):
        """
        Constructor of the Tnet.

        :param k: Number of input features.
        :type k: int
        """
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self,
                _input: torch.Tensor):
        """
        Forward pass of the Tnet.

        :param input: Input tensor of shape (batch_size, k, n)
        :type input: torch.Tensor

        :return: Output matrix of shape (batch_size, k, k)
        :rtype: torch.Tensor
        """
        batch_size = _input.size(0)
        x = self.relu(self.bn1(self.conv1(_input)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        pool = nn.MaxPool1d(x.size(-1))(x)
        flat = nn.Flatten(1)(pool)
        x = self.relu(self.bn4(self.fc1(flat)))
        x = self.relu(self.bn5(self.fc2(x)))

        init = torch.eye(self.k, requires_grad=True).repeat(
            batch_size, 1, 1)
        if x.is_cuda:
            init = init.cuda()
        matrix = self.fc3(x).view(-1, self.k, self.k) + init
        return matrix


class PointNet(nn.Module):
    """
    PointNet architecture.

    See details in https://arxiv.org/pdf/1612.00593
    """

    def __init__(self,
                 num_classes: int = 10):
        """
        Constructor of the PointNet.

        :param num_classes: Number of classes.
        :type num_classes: int
        """
        super(PointNet, self).__init__()
        self.tnet3 = Tnet(k=3)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.tnet64 = Tnet(k=64)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)
        self.softmax = nn.LogSoftmax(dim=1)

        self.relu = nn.ReLU()

    def forward(self,
                _input: torch.Tensor):
        """
        Forward pass of the PointNet.

        :param input: Input tensor of shape (batch_size, 3, n)
        :type input: torch.Tensor
        """
        t3 = self.tnet3(_input)
        x = torch.bmm(torch.transpose(_input, 1, 2), t3).transpose(1, 2)
        x = self.relu(self.bn1(self.conv1(x)))
        t64 = self.tnet64(x)
        x = torch.bmm(torch.transpose(x, 1, 2), t64).transpose(1, 2)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = nn.MaxPool1d(x.size(-1))(x)
        x = nn.Flatten(1)(x)
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        output = self.softmax(x)
        return output
    
    def loss(self,
             outputs: torch.Tensor,
             labels: torch.Tensor,
             regularize_weight: float = 0.001):
        """
        Loss function of the PointNet.

        :param outputs: Output tensor of shape (batch_size, num_classes)
        :type outputs: torch.Tensor
        :param labels: Label tensor of shape (batch_size)
        :type labels: torch.Tensor
        :param regularize_weight: Regularization weight.
        :type regularize_weight: float
        """
        criterion = torch.nn.NLLLoss()
        batch_size = outputs.size(0)
        i3 = torch.eye(3, requires_grad=True).repeat(
            batch_size, 1, 1)
        i64 = torch.eye(64, requires_grad=True).repeat(
            batch_size, 1, 1)
        if outputs.is_cuda:
            i3 = i3.cuda()
            i64 = i64.cuda()
        diff3 = i3-torch.bmm(
            self.tnet3.fc3.weight.view(-1, 3, 3),
            self.tnet3.fc3.weight.view(-1, 3, 3).transpose(1, 2))
        diff64 = i64-torch.bmm(
            self.tnet64.fc3.weight.view(-1, 64, 64),
            self.tnet64.fc3.weight.view(-1, 64, 64).transpose(1, 2))
        return criterion(outputs, labels) + regularize_weight * (
            torch.norm(diff3)+torch.norm(diff64)) / float(batch_size)
