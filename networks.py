import torch
import torch.nn as nn
import torch.nn.functional as F

class DNN(nn.Module):
    def __init__(self, input_n):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_n, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 3)
    
    def forward(self, x):
        model = self.fc1(x)
        model = self.relu(model)
        model = self.fc2(model)
        return model 

