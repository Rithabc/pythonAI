import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# import random
# import sys
# sys.setrecursionlimit(10000)

# def generate_random_grid(size):
#     grid = [[1 for _ in range(size)] for _ in range(size)]
    
#     start_row = random.randint(0, size - 1)
#     start_col = random.randint(0, size - 1)
    
#     def dfs(row, col):
#         z = 0
#         if row < 0 or row >= size or col < 0 or col >= size:
#             return
#         if grid[row][col] == 0:
#             return
        
#         if row - 1 >= 0 and row - 1 < size:
#             if grid[row - 1][col] == 0:
#                 z += 1
#                 if z > 1:
#                     return
        
#         if row + 1 >= 0 and row + 1 < size:
#             if grid[row + 1][col] == 0:
#                 z += 1
#                 if z > 1:
#                     return
        
#         if col + 1 >= 0 and col + 1 < size:
#             if grid[row][col + 1] == 0:
#                 z += 1
#                 if z > 1:
#                     return
        
#         if col - 1 >= 0 and col - 1 < size:
#             if grid[row][col - 1] == 0:
#                 z += 1
#                 if z > 1:
#                     return
        
#         if z > 1:
#             return
        
#         grid[row][col] = 0
        
#         cell = random.randint(0, 3)
#         if cell == 0:
#             dfs(row + 1, col)
#             dfs(row - 1, col)
#             dfs(row, col - 1)
#             dfs(row, col + 1)
#         elif cell == 1:
#             dfs(row - 1, col)
#             dfs(row + 1, col)
#             dfs(row, col + 1)
#             dfs(row, col - 1)
#         elif cell == 2:
#             dfs(row, col - 1)
#             dfs(row, col + 1)
#             dfs(row + 1, col)
#             dfs(row - 1, col)
#         else:
#             dfs(row, col + 1)
#             dfs(row, col - 1)
#             dfs(row + 1, col)
#             dfs(row - 1, col)
    
#     dfs(start_row, start_col)
#     return grid

# size = int(input("Enter grid size: "))
# grid = generate_random_grid(size)

# print("\nGenerated grid:")
# for row in grid:
#     print(''.join(map(str, row)))

# with open('input_grid.txt', 'w') as f:
#     f.write(f"{size}\n")
#     for row in grid:
#         f.write(''.join(map(str, row)) + '\n')

# print("\nGrid saved to input_grid.txt")



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

X = np.array(features)
y = np.array(labels)

print(f"Total samples: {len(X)}")
if len(X) < 50:
    print("WARNING: Too few samples! Generate more data in C++ (increase loop iterations)")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_split=2, random_state=42)
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print(f"R² Score: {score:.3f}")
# print(f"Training samples: {len(X_train)}")
# print(f"Test samples: {len(X_test)}")
# print(f"\nLabel stats:")
# print(f"  Min moves: {y.min()}")
# print(f"  Max moves: {y.max()}")
# print(f"  Mean moves: {y.mean():.1f}")
# print(f"  Std moves: {y.std():.1f}")
# print(f"\nFeature ranges:")
# for i, name in enumerate(['size', 'obstacles', 'density', 'free_cells']):
    # print(f"  {name}: {X[:, i].min():.1f} - {X[:, i].max():.1f}")



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
    
    input_features = [[size, num_obstacles, obstacle_density, num_bots]]
    predicted_moves = model.predict(input_features)
    print(f"\nPrediction for input grid (from input_grid.txt):")
    print(f"  Features: size={size}, obstacles={num_obstacles}, density={obstacle_density:.2f}, free={num_bots}")
    print(f"  Predicted optimal moves: {predicted_moves[0]:.0f}")
except FileNotFoundError:
    print("\nNo input_grid.txt found. Run generate_grid.py first.")