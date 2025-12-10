import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import xgboost as xgb
import joblib

# ================================
# Model definitions
# ================================
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim=4):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    def forward(self, x):
        return self.linear(x)

class MLPModel(nn.Module):
    def __init__(self, input_dim=4):
        super(MLPModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.layers(x)

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

# ================================
# Load and preprocess data
# ================================
def load_data(file_path='grid_maps.txt'):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    grids, labels = [], []
    max_size = 0
    i = 0
    while i < len(lines):
        size = int(lines[i].strip())
        max_size = max(max_size, size)
        grid = []
        for j in range(1, size + 1):
            grid.append([int(c) for c in lines[i + j].strip()])
        moves = int(lines[i + size + 1].strip())
        grids.append(np.array(grid))
        labels.append(moves)
        i += size + 3
    return grids, labels, max_size

# ================================
# Feature extraction for Linear/MLP/XGBoost
# ================================
def extract_features(grids):
    features = []
    for g in grids:
        size = g.shape[0]
        num_obstacles = np.sum(g)
        obstacle_density = num_obstacles / (size*size)
        num_bots = size*size - num_obstacles
        features.append([size, num_obstacles, obstacle_density, num_bots])
    return np.array(features, dtype=np.float32)

# ================================
# Training functions
# ================================
def train_linear(X_train, y_train, lr=0.01, epochs=100):
    model = LinearRegressionModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    for _ in range(epochs):
        optimizer.zero_grad()
        output = model(X_train_t)
        loss = criterion(output, y_train_t)
        loss.backward()
        optimizer.step()
    return model

def train_mlp(X_train, y_train, lr=0.01, epochs=200):
    model = MLPModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_train_t)
        loss = criterion(output, y_train_t)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    return model

def train_cnn(grids, labels, max_size, lr=0.01, epochs=200):
    padded_grids = []
    for g in grids:
        pad = max_size - g.shape[0]
        g_pad = np.pad(g, ((0,pad),(0,pad)), 'constant', constant_values=1)
        padded_grids.append(g_pad)
    X = np.array(padded_grids)[:, None, :, :]
    y = np.array(labels, dtype=np.float32)
    model = CNNModel(max_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()
    X_t = torch.FloatTensor(X)
    y_t = torch.FloatTensor(y).unsqueeze(1)
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_t)
        loss = criterion(output, y_t)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 20 == 0:
            print(f"CNN Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    return model

# ================================
# Train XGBoost
# ================================
def train_xgb(X_train, y_train):
    model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.01, random_state=42)
    model.fit(X_train, y_train)
    return model

# ================================
# Prediction function for input_grid.txt
# ================================
def predict_input_grid(linear_model, mlp_model, xgb_model, cnn_model=None, cnn_max_size=None, input_file='input_grid.txt'):
    try:
        with open(input_file, 'r') as f:
            lines = f.readlines()
        size = int(lines[0].strip())
        input_grid = []
        for i in range(1, size + 1):
            input_grid.append([int(c) for c in lines[i].strip()])
        flat_grid = [cell for row in input_grid for cell in row]
        num_obstacles = sum(flat_grid)
        obstacle_density = num_obstacles / len(flat_grid)
        num_bots = len(flat_grid) - num_obstacles
        
        print(f"\nInput grid analysis:")
        print(f"  Size: {size}x{size}")
        print(f"  Obstacles: {num_obstacles}")
        print(f"  Free cells: {num_bots}")
        print(f"  Density: {obstacle_density:.2f}")
        
        if num_bots == 0:
            print("\n⚠️ INVALID GRID: No free cells. Cannot predict moves.")
            return
        elif num_bots == 1:
            print("\n✓ Only 1 free cell. Optimal moves: 0")
            return

        # Features
        input_features = torch.FloatTensor([[size, num_obstacles, obstacle_density, num_bots]])
        input_features_np = np.array([[size, num_obstacles, obstacle_density, num_bots]])

        if cnn_model is not None:
            padded_grid = np.pad(np.array(input_grid), ((0, cnn_max_size - size), (0, cnn_max_size - size)), 'constant', constant_values=1)
            cnn_input = torch.FloatTensor(padded_grid).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            linear_pred = linear_model(input_features)
            mlp_pred = mlp_model(input_features)
            xgb_pred = xgb_model.predict(input_features_np)
            if cnn_model is not None:
                cnn_pred = cnn_model(cnn_input)
        
        print(f"\nPredictions:")
        print(f"  Linear Regression: {linear_pred.item():.0f}")
        print(f"  XGBoost: {xgb_pred[0]:.0f}")
        print(f"  MLP: {mlp_pred.item():.0f}")
        if cnn_model is not None:
            print(f"  CNN: {cnn_pred.item():.0f}")
        
    except FileNotFoundError:
        print("\nNo input_grid.txt found.")

# ================================
# Main
# ================================
if __name__ == '__main__':
    grids, labels, max_size = load_data('grid_maps.txt')
    features = extract_features(grids)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    print("\n=== Training Linear Regression ===")
    linear_model = train_linear(X_train, y_train)
    torch.save(linear_model.state_dict(), 'model_linear.pth')

    print("\n=== Training MLP ===")
    mlp_model = train_mlp(X_train, y_train)
    torch.save(mlp_model.state_dict(), 'model_mlp.pth')

    print("\n=== Training XGBoost ===")
    xgb_model = train_xgb(X_train, y_train)
    joblib.dump(xgb_model, 'model_xgb.pkl')

    print("\n=== Training CNN ===")
    cnn_model = train_cnn(grids, labels, max_size)
    torch.save(cnn_model.state_dict(), 'model_cnn.pth')

    print("\n=== Predicting input_grid.txt ===")
    predict_input_grid(linear_model, mlp_model, xgb_model, cnn_model, max_size)
