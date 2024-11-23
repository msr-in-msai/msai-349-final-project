import torch
import torch.nn as nn

class PointNet(nn.Module):

    def __init__(self):
        
        super(PointNet, self).__init__()

        self.t_net3x3 = nn.Conv1d(3,3,kernel_size=1)
        self.mlp1_1 = nn.Linear(3,64)
        self.mlp1_2 = nn.Linear(64,64)
        self.t_net64x64 = nn.Conv1d(64,64,kernel_size=1)
        self.mlp2_1 = nn.Linear(64,128)
        self.mlp2_2 = nn.Linear(128,1024)
    
    def forward(self,x):
        
        x = self.t_net3x3(x)

        x = x.view(-1,3)
        x = self.mlp1_1(x)
        x = self.mlp1_2(x)

        x = (x.T).unsqueeze(0)
        x = self.t_net64x64(x)

        x = x.view(-1,64)
        x = self.mlp2_1(x)
        x = self.mlp2_2(x)
        
        # Max pooling


        return x

p = PointNet()
input_data = torch.randn(1,3,10)
# print(input_data)
print(p(input_data).shape)