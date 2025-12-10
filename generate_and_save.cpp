#include <iostream>
#include <vector>
#include <fstream>
#include <random>
#include <chrono>

using namespace std;

mt19937 gen(chrono::system_clock::now().time_since_epoch().count());

int generateRandom(int low, int high) {
    uniform_int_distribution<> dist(low, high);
    return dist(gen);
}

void dfs(int row, int col, int n, vector<vector<int>> &grid) {
    int z = 0;
    if (row < 0 || row >= n || col < 0 || col >= n)
        return;
    if (grid[row][col] == 0)
        return;
    if (row - 1 >= 0 && row - 1 < n) {
        if (grid[row - 1][col] == 0) {
            z++;
            if (z > 1)
                return;
        }
    }
    if (row + 1 >= 0 && row + 1 < n) {
        if (grid[row + 1][col] == 0) {
            z++;
            if (z > 1)
                return;
        }
    }
    if (col + 1 >= 0 && col + 1 < n) {
        if (grid[row][col + 1] == 0) {
            z++;
            if (z > 1)
                return;
        }
    }
    if (col - 1 >= 0 && col - 1 < n) {
        if (grid[row][col - 1] == 0) {
            z++;
            if (z > 1)
                return;
        }
    }

    if (z > 1)
        return;
    grid[row][col] = 0;

    int cell = generateRandom(0, 3);
    switch (cell) {
    case 0:
        dfs(row + 1, col, n, grid);
        dfs(row - 1, col, n, grid);
        dfs(row, col - 1, n, grid);
        dfs(row, col + 1, n, grid);
        break;
    case 1:
        dfs(row - 1, col, n, grid);
        dfs(row + 1, col, n, grid);
        dfs(row, col + 1, n, grid);
        dfs(row, col - 1, n, grid);
        break;
    case 2:
        dfs(row, col - 1, n, grid);
        dfs(row, col + 1, n, grid);
        dfs(row + 1, col, n, grid);
        dfs(row - 1, col, n, grid);
        break;
    case 3:
        dfs(row, col + 1, n, grid);
        dfs(row, col - 1, n, grid);
        dfs(row + 1, col, n, grid);
        dfs(row - 1, col, n, grid);
        break;
    }
}

vector<vector<int>> generateGrid(int n) {
    vector<vector<int>> grid(n, vector<int>(n, 1));
    int ind = generateRandom(0, n - 1);
    dfs(ind, ind, n, grid);
    return grid;
}

int main() {
    int n;
    cout << "Enter grid size: ";
    cin >> n;
    
    vector<vector<int>> grid = generateGrid(n);
    
    ofstream outFile("input_grid.txt");
    outFile << n << "\n";
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            outFile << grid[i][j];
        }
        outFile << "\n";
    }
    outFile.close();
    
    cout << "Grid saved to input_grid.txt\n";
    return 0;
}
