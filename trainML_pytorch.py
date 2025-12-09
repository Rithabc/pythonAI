import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

with open('grid_maps.txt', 'r') as f:
    lines = f.readlines()

grids = []
labels = []
i = 0
while i < len(lines):
    size = int(lines[i].strip())
    grid = []
    for j in range(1, size + 1):
        grid.extend([int(c) for c in lines[i + j].strip()])
    moves = int(lines[i + size + 1].strip())
    grids.append(grid)
    labels.append(moves)
    i += size + 3

features = []
for grid in grids:
    size = int(np.sqrt(len(grid)))
    g = np.array(grid[:size*size])
    features.append([
        size,
        np.sum(g),
        np.sum(g) / len(g),
        len(g) - np.sum(g)
    ])

X = np.array(features, dtype=np.float32)
y = np.array(labels, dtype=np.float32).reshape(-1, 1)

print(f"Total samples: {len(X)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train)
y_test = torch.FloatTensor(y_test)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

model = MLP()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 200
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

model.eval()
with torch.no_grad():
    predictions = model(X_test)
    mse = criterion(predictions, y_test)
    print(f'\nTest MSE: {mse.item():.4f}')

torch.save(model.state_dict(), 'model_pytorch.pth')
print("Model saved to model_pytorch.pth")

try:
    with open('input_grid.txt', 'r') as f:
        lines = f.readlines()
    size = int(lines[0].strip())
    input_grid = []
    for i in range(1, size + 1):
        input_grid.append([int(c) for c in lines[i].strip()])
    
    flat_grid = [cell for row in input_grid for cell in row]
    num_obstacles = sum(flat_grid)
    obstacle_density = num_obstacles / len(flat_grid)
    num_bots = len(flat_grid) - num_obstacles
    
    input_features = torch.FloatTensor([[size, num_obstacles, obstacle_density, num_bots]])
    with torch.no_grad():
        predicted_moves = model(input_features)
    print(f"\nPrediction for input grid:")
    print(f"  Features: size={size}, obstacles={num_obstacles}, density={obstacle_density:.2f}, free={num_bots}")
    print(f"  Predicted optimal moves: {predicted_moves.item():.0f}")
except FileNotFoundError:
    print("\nNo input_grid.txt found.")
