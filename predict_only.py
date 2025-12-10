import numpy as np
import torch
import torch.nn as nn
import joblib

class CNNModel(nn.Module):
    def __init__(self, grid_size):
        super(CNNModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * grid_size * grid_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)

with open('input_grid.txt', 'r') as f:
    lines = f.readlines()
size = int(lines[0].strip())
grid = []
for i in range(1, size + 1):
    grid.append([int(c) for c in lines[i].strip()])

checkpoint = torch.load('model_cnn.pth')
fc_weight_shape = checkpoint['fc_layers.0.weight'].shape[1]
max_size = int(np.sqrt(fc_weight_shape / 32))

cnn_model = CNNModel(max_size)
cnn_model.load_state_dict(checkpoint)
cnn_model.eval()

padded_grid = np.pad(np.array(grid), ((0, max_size - size), (0, max_size - size)), 'constant', constant_values=1)
cnn_input = torch.FloatTensor(padded_grid).unsqueeze(0).unsqueeze(0)

with torch.no_grad():
    cnn_pred = cnn_model(cnn_input)

print(f"CNN: {int(cnn_pred.item())}")
