import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import xgboost as xgb
import joblib

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

X_train_torch = torch.FloatTensor(X_train)
X_test_torch = torch.FloatTensor(X_test)
y_train_torch = torch.FloatTensor(y_train)
y_test_torch = torch.FloatTensor(y_test)

class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(4, 1)
    
    def forward(self, x):
        return self.linear(x)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

print("\n=== Training Linear Regression ===")
linear_model = LinearRegression()
criterion = nn.L1Loss()
optimizer_linear = torch.optim.Adam(linear_model.parameters(), lr=0.01)

for epoch in range(100):
    linear_model.train()
    optimizer_linear.zero_grad()
    outputs = linear_model(X_train_torch)
    loss = criterion(outputs, y_train_torch)
    loss.backward()
    optimizer_linear.step()

linear_model.eval()
with torch.no_grad():
    predictions = linear_model(X_test_torch)
    mse = criterion(predictions, y_test_torch)
    print(f'Linear Regression Test MSE: {mse.item():.4f}')

torch.save(linear_model.state_dict(), 'model_linear_pytorch.pth')

print("\n=== Training XGBoost ===")
xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train.ravel())
y_pred_xgb = xgb_model.predict(X_test)

mse_xgb = np.mean((y_pred_xgb - y_test.ravel())**2)
rmse_xgb = np.sqrt(mse_xgb)
mae_xgb = np.mean(np.abs(y_pred_xgb - y_test.ravel()))

print(f'XGBoost - MSE: {mse_xgb:.4f}, RMSE: {rmse_xgb:.4f}, MAE: {mae_xgb:.4f}')
joblib.dump(xgb_model, 'model_xgboost.pkl')

print("\n=== Training MLP ===")
model = MLP()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 200
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_torch)
    loss = criterion(outputs, y_train_torch)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

model.eval()
with torch.no_grad():
    predictions = model(X_test_torch)
    mse = criterion(predictions, y_test_torch)
    print(f'\nMLP Test MSE: {mse.item():.4f}')

torch.save(model.state_dict(), 'model_pytorch.pth')
print("Models saved")

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
    input_features_np = np.array([[size, num_obstacles, obstacle_density, num_bots]])
    
    with torch.no_grad():
        linear_pred = linear_model(input_features)
        mlp_pred = model(input_features)
    xgb_pred = xgb_model.predict(input_features_np)
    
    print(f"\nPrediction for input grid:")
    print(f"  Features: size={size}, obstacles={num_obstacles}, density={obstacle_density:.2f}, free={num_bots}")
    print(f"  Linear Regression: {linear_pred.item():.0f}")
    print(f"  XGBoost: {xgb_pred[0]:.0f}")
    print(f"  MLP: {mlp_pred.item():.0f}")
except FileNotFoundError:
    print("\nNo input_grid.txt found.")
