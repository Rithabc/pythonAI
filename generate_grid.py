import random
import sys
sys.setrecursionlimit(10000)

def generate_random_grid(size):
    grid = [[1 for _ in range(size)] for _ in range(size)]
    
    start_row = random.randint(0, size - 1)
    start_col = random.randint(0, size - 1)
    
    def dfs(row, col):
        z = 0
        if row < 0 or row >= size or col < 0 or col >= size:
            return
        if grid[row][col] == 0:
            return
        
        if row - 1 >= 0 and row - 1 < size:
            if grid[row - 1][col] == 0:
                z += 1
                if z > 1:
                    return
        
        if row + 1 >= 0 and row + 1 < size:
            if grid[row + 1][col] == 0:
                z += 1
                if z > 1:
                    return
        
        if col + 1 >= 0 and col + 1 < size:
            if grid[row][col + 1] == 0:
                z += 1
                if z > 1:
                    return
        
        if col - 1 >= 0 and col - 1 < size:
            if grid[row][col - 1] == 0:
                z += 1
                if z > 1:
                    return
        
        if z > 1:
            return
        
        grid[row][col] = 0
        
        cell = random.randint(0, 3)
        if cell == 0:
            dfs(row + 1, col)
            dfs(row - 1, col)
            dfs(row, col - 1)
            dfs(row, col + 1)
        elif cell == 1:
            dfs(row - 1, col)
            dfs(row + 1, col)
            dfs(row, col + 1)
            dfs(row, col - 1)
        elif cell == 2:
            dfs(row, col - 1)
            dfs(row, col + 1)
            dfs(row + 1, col)
            dfs(row - 1, col)
        else:
            dfs(row, col + 1)
            dfs(row, col - 1)
            dfs(row + 1, col)
            dfs(row - 1, col)
    
    dfs(start_row, start_col)
    return grid

size = int(input("Enter grid size: "))
grid = generate_random_grid(size)

print("\nGenerated grid:")
for row in grid:
    print(''.join(map(str, row)))

with open('input_grid.txt', 'w') as f:
    f.write(f"{size}\n")
    for row in grid:
        f.write(''.join(map(str, row)) + '\n')

print("\nGrid saved to input_grid.txt")
