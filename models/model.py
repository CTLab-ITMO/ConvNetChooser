import torch
import torch.nn as nn
import torch.nn.functional as F

# Определение параметров модели
class TwoLayerClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(TwoLayerClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = F.softmax(out, dim=1)
        return out