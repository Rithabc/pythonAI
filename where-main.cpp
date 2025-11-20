#include <iostream>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <queue>
#include <stack>
#include <set>
#include <cstring>
#include <cfloat>
#include <ctime>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <math.h>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <random>
#include <unordered_map>
#include <queue>
#include <stack>
#include <set>

#include <cstring>

using namespace std;

std::mt19937 gen(std::chrono::system_clock::now().time_since_epoch().count());


typedef pair<int, int> Pair;
typedef pair<double, pair<int, int>> pPair;

namespace std {
    template <>
    struct hash<std::pair<int, int>> {
        std::size_t operator()(const std::pair<int, int> &p) const noexcept {
            return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
        }
    };
}

struct cell {
    int parent_i, parent_j;
    double f, g, h;
};

int manhattan(int r1, int c1, int r2, int c2) {
    return abs(r1 - r2) + abs(c1 - c2);
}

bool isValid(int row, int col, int size) {
    return (row >= 0) && (row < size) && (col >= 0) && (col < size);
}

bool isUnBlocked(int **grid, int row, int col) {
    return grid[row][col] == 0;
}

bool isDestination(int row, int col, Pair dest) {
    return (row == dest.first && col == dest.second);
}

double calculateHValue(int row, int col, Pair dest) {
    return sqrt((row - dest.first) * (row - dest.first) + 
                (col - dest.second) * (col - dest.second));
}

vector<Pair> edit;
vector<string> directions;

void tracePath(vector<vector<cell>> cellDetails, Pair dest) {
    int row = dest.first;
    int col = dest.second;
    stack<Pair> Path;

    while (!(cellDetails[row][col].parent_i == row && 
             cellDetails[row][col].parent_j == col)) {
        Path.push(make_pair(row, col));
        int temp_row = cellDetails[row][col].parent_i;
        int temp_col = cellDetails[row][col].parent_j;
        row = temp_row;
        col = temp_col;
    }

    int tr = row;
    int tc = col;
    Path.push(make_pair(row, col));
    
    while (!Path.empty()) {
        pair<int, int> p = Path.top();
        Path.pop();
        
        if (p.first == tr - 1 && p.second == tc) {
            edit.push_back(make_pair(-1, 0));
            directions.push_back("UP");
        } else if (p.first == tr + 1 && p.second == tc) {
            edit.push_back(make_pair(1, 0));
            directions.push_back("DOWN");
        } else if (p.first == tr && p.second == tc - 1) {
            edit.push_back(make_pair(0, -1));
            directions.push_back("LEFT");
        } else if (p.first == tr && p.second == tc + 1) {
            edit.push_back(make_pair(0, 1));
            directions.push_back("RIGHT");
        }
        tr = p.first;
        tc = p.second;
    }
}

void aStarSearch(int **grid, Pair src, Pair dest, int size) {
    if (!isValid(src.first, src.second, size) || 
        !isValid(dest.first, dest.second, size)) {
        return;
    }

    if (!isUnBlocked(grid, src.first, src.second) || 
        !isUnBlocked(grid, dest.first, dest.second)) {
        return;
    }

    if (isDestination(src.first, src.second, dest)) {
        return;
    }

    bool closedList[size][size];
    memset(closedList, false, sizeof(closedList));

    vector<vector<cell>> cellDetails(size, vector<cell>(size));

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            cellDetails[i][j].f = FLT_MAX;
            cellDetails[i][j].g = FLT_MAX;
            cellDetails[i][j].h = FLT_MAX;
            cellDetails[i][j].parent_i = -1;
            cellDetails[i][j].parent_j = -1;
        }
    }

    int i = src.first, j = src.second;
    cellDetails[i][j].f = 0.0;
    cellDetails[i][j].g = 0.0;
    cellDetails[i][j].h = 0.0;
    cellDetails[i][j].parent_i = i;
    cellDetails[i][j].parent_j = j;

    set<pPair> openList;
    openList.insert(make_pair(0.0, make_pair(i, j)));

    while (!openList.empty()) {
        pPair p = *openList.begin();
        openList.erase(openList.begin());

        i = p.second.first;
        j = p.second.second;
        closedList[i][j] = true;

        int directions[4][2] = {{-1, 0}, {1, 0}, {0, 1}, {0, -1}};
        
        for (int d = 0; d < 4; d++) {
            int newI = i + directions[d][0];
            int newJ = j + directions[d][1];

            if (isValid(newI, newJ, size)) {
                if (isDestination(newI, newJ, dest)) {
                    cellDetails[newI][newJ].parent_i = i;
                    cellDetails[newI][newJ].parent_j = j;
                    tracePath(cellDetails, dest);
                    return;
                } else if (!closedList[newI][newJ] && isUnBlocked(grid, newI, newJ)) {
                    double gNew = cellDetails[i][j].g + 1.0;
                    double hNew = calculateHValue(newI, newJ, dest);
                    double fNew = gNew + hNew;

                    if (cellDetails[newI][newJ].f == FLT_MAX || 
                        cellDetails[newI][newJ].f > fNew) {
                        openList.insert(make_pair(fNew, make_pair(newI, newJ)));
                        cellDetails[newI][newJ].f = fNew;
                        cellDetails[newI][newJ].g = gNew;
                        cellDetails[newI][newJ].h = hNew;
                        cellDetails[newI][newJ].parent_i = i;
                        cellDetails[newI][newJ].parent_j = j;
                    }
                }
            }
        }
    }
}

float sensorModel(Pair locator, Pair roomba, float alpha, bool beep) {
    int dist = manhattan(locator.first, locator.second, roomba.first, roomba.second);
    float p_beep = exp(-alpha * (dist - 1));
    return beep ? p_beep : (1.0 - p_beep);
}

int simulateBeep(Pair locator, Pair trueRoombaPos, float alpha) {
    if (locator.first == trueRoombaPos.first && 
        locator.second == trueRoombaPos.second) {
        return 1;
    }
    
    int dist = manhattan(locator.first, locator.second, 
                        trueRoombaPos.first, trueRoombaPos.second);
    float p_beep = exp(-alpha * (dist - 1));
    
    float r = (float)rand() / RAND_MAX;
    return (r < p_beep) ? 0 : -1;
}

void updateBelief(unordered_map<Pair, float>& belief, 
                  Pair locatorPos, 
                  bool beepHeard, 
                  float alpha,
                  bool foundExact = false) {
    
    if (foundExact) {
        for (auto& entry : belief) {
            entry.second = (entry.first == locatorPos) ? 1.0 : 0.0;
        }
        return;
    }
    
    float normalization = 0.0;
    
    for (auto& entry : belief) {
        Pair roombaPos = entry.first;
        float prior = entry.second;
        float likelihood = sensorModel(locatorPos, roombaPos, alpha, beepHeard);
        entry.second = likelihood * prior;
        normalization += entry.second;
    }
    
    for (auto& entry : belief) {
        entry.second /= normalization;
    }
}

Pair getMaxProbabilityCell(const unordered_map<Pair, float>& belief) {
    Pair maxCell = belief.begin()->first;
    float maxProb = 0.0;
    
    for (const auto& entry : belief) {
        if (entry.second > maxProb) {
            maxProb = entry.second;
            maxCell = entry.first;
        }
    }
    
    return maxCell;
}

void updateBeliefAfterMove(unordered_map<Pair, float>& belief, 
                          string move, int** grid, int size) {
    unordered_map<Pair, float> newBelief;
    
    for (const auto& entry : belief) {
        Pair currentPos = entry.first;
        float prob = entry.second;
        
        int newI = currentPos.first;
        int newJ = currentPos.second;
        
        if (move == "UP") newI--;
        else if (move == "DOWN") newI++;
        else if (move == "LEFT") newJ--;
        else if (move == "RIGHT") newJ++;
        
        if (isValid(newI, newJ, size) && isUnBlocked(grid, newI, newJ)) {
            Pair newPos = make_pair(newI, newJ);
            newBelief[newPos] += prob;
        } else {
            newBelief[currentPos] += prob;
        }
    }
    
    belief = newBelief;
}



vector<int> baselineStrategy2_trueOptimized2(int **grid, int size, Pair locatorStart, 
                                     Pair trueRoombaPos, float alpha, 
                                     vector<Pair> allOpenCells,
                                     const vector<vector<int>>& gridVec) {
                                        vector<int> ans;

    unordered_map<Pair, float> belief;
    for (auto cell : allOpenCells) {
        belief[cell] = 1.0 / allOpenCells.size();
    }

    Pair locator = locatorStart;
    Pair roombaPos = trueRoombaPos;
    int totalActions = 0;
    int senseCount = 0;
    int moveCount = 0;
    bool roombaFound = false;
    set<Pair> visitedCells;

    while (!roombaFound && totalActions < 50) {
        int beepResult = simulateBeep(locator, roombaPos, alpha);
        totalActions++;
        senseCount++;

        if (beepResult == 1) {
            roombaFound = true;
            break;
        }

        bool beepHeard = (beepResult == 0);
        updateBelief(belief, locator, beepHeard, alpha, false);
        visitedCells.insert(locator);

        Pair maxProbCell = getMaxProbabilityCell(belief);
        
        // If we're at max prob cell and have visited it before, try second best
        if (locator.first == maxProbCell.first && locator.second == maxProbCell.second) {
            // Find unvisited cell with highest probability
            Pair bestUnvisited = maxProbCell;
            float bestProb = -1.0;
            for (const auto& entry : belief) {
                if (visitedCells.find(entry.first) == visitedCells.end() && entry.second > bestProb) {
                    bestProb = entry.second;
                    bestUnvisited = entry.first;
                }
            }
            
            // If all cells visited, reset visited set (except current)
            if (bestProb < 0) {
                visitedCells.clear();
                visitedCells.insert(locator);
                bestUnvisited = maxProbCell;
                for (const auto& entry : belief) {
                    if (entry.first != locator && entry.second > bestProb) {
                        bestProb = entry.second;
                        bestUnvisited = entry.first;
                    }
                }
            }
            
            maxProbCell = bestUnvisited;
        }

        if (locator.first == maxProbCell.first && locator.second == maxProbCell.second) {
            continue;
        }

        directions.clear();
        edit.clear();
        aStarSearch(grid, locator, maxProbCell, size);

        if (directions.empty()) {
            continue;
        }

        string nextMove = directions[0];
        totalActions++;
        moveCount++;

        int newI = locator.first;
        int newJ = locator.second;
        
        if (nextMove == "UP") newI--;
        else if (nextMove == "DOWN") newI++;
        else if (nextMove == "LEFT") newJ--;
        else if (nextMove == "RIGHT") newJ++;

        if (isValid(newI, newJ, size) && isUnBlocked(grid, newI, newJ)) {
            locator.first = newI;
            locator.second = newJ;
        }
    }

    // cout << senseCount << " " << moveCount << " " << totalActions << endl;
    ans.push_back(senseCount);
    ans.push_back(moveCount);
    ans.push_back(totalActions);

    return ans;
}

// std::mt19937 gen(std::chrono::system_clock::now().time_since_epoch().count());

int generateRandom(int low, int high)
{

    std::uniform_int_distribution<> dist(low, high);
    return dist(gen);
}

void dfs(int row, int col, int n, vector<vector<int>> &grid)
{
    int z = 0;
    if (row < 0 || row >= n || col < 0 || col >= n)
        return;
    if (grid[row][col] == 0)
        return;
    if (row - 1 >= 0 && row - 1 < n)
    {
        if (grid[row - 1][col] == 0)
        {
            z++;
            if (z > 1)
                return;
        }
    }
    if (row + 1 >= 0 && row + 1 < n)
    {
        if (grid[row + 1][col] == 0)
        {
            z++;
            if (z > 1)
                return;
        }
    }
    if (col + 1 >= 0 && col + 1 < n)
    {
        if (grid[row][col + 1] == 0)
        {
            z++;
            if (z > 1)
                return;
        }
    }
    if (col - 1 >= 0 && col - 1 < n)
    {
        if (grid[row][col - 1] == 0)
        {
            z++;
            if (z > 1)
                return;
        }
    }

    if (z > 1)
        return;
    grid[row][col] = 0;

    int cell = generateRandom(0, 3);
    switch (cell)
    {
    case 0:
    {
        dfs(row + 1, col, n, grid);
        dfs(row - 1, col, n, grid);
        dfs(row, col - 1, n, grid);
        dfs(row, col + 1, n, grid);
        break;
    }
    case 1:
    {
        dfs(row - 1, col, n, grid);
        dfs(row + 1, col, n, grid);
        dfs(row, col + 1, n, grid);
        dfs(row, col - 1, n, grid);
        break;
    }
    case 2:
    {
        dfs(row, col - 1, n, grid);
        dfs(row, col + 1, n, grid);
        dfs(row + 1, col, n, grid);
        dfs(row - 1, col, n, grid);
        break;
    }
    case 3:
    {
        dfs(row, col + 1, n, grid);
        dfs(row, col - 1, n, grid);
        dfs(row + 1, col, n, grid);
        dfs(row - 1, col, n, grid);
        break;
    }
    }
}

vector<vector<int>> generateGrid(int n)
{
    vector<vector<int>> grid(n, vector<int>(n, 1));
    int ind = generateRandom(0, n - 1);
    cout << ind << endl;
    // grid[ind][ind] = 0;
    dfs(ind, ind, n, grid);

    for (int row = 0; row < n; row++)
    {
        for (int col = 0; col < n; col++)
        {
            // cout << grid[row][col] << " ";
        }
        // cout << endl;
    }

    return grid;
}

int main() {
    srand(time(0));


    
    vector<vector<int>> gridVec = generateGrid(5);


    
    int n = gridVec.size();
    
    int** grid = new int*[n];
    for (int i = 0; i < n; i++) {
        grid[i] = new int[n];
        for (int j = 0; j < n; j++) {
            // cout << gridVec[i][j] << " ";
            grid[i][j] = gridVec[i][j];
        }
        // cout << endl;
    }
    
    vector<Pair> allOpenCells;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (grid[i][j] == 0) {
                allOpenCells.push_back(make_pair(i, j));
            }
        }
    }
    
    Pair locatorStart = allOpenCells[0];
    Pair trueRoombaPos = allOpenCells[generateRandom(0, allOpenCells.size() - 1)];
    // float alpha = 0.5;
    cout << "Sense" << " + Move" << " = Total Actions\n";
    vector< vector<int> > alphaResults(26, vector<int>(3,0));
    for(int i = 0; i < 10; i++) {

    for(float alpha = 0.1; alpha <= 1.0; alpha += 0.1) {
        // cout << alpha << " ";
         vector<int> totalAction = baselineStrategy2_trueOptimized2(grid, n, locatorStart, 
            trueRoombaPos, alpha, 
            allOpenCells, gridVec);
            // vector<int> results = alphaResults[alpha];
            // results[0] += totalAction[0];
            // results[1] += totalAction[1];
            // results[2] += totalAction[2];
            // alphaResults[alpha] = results;

            alphaResults[(int)(alpha*10)-1][0] += totalAction[0];
            alphaResults[(int)(alpha*10)-1][1] += totalAction[1];
            alphaResults[(int)(alpha*10)-1][2] += totalAction[2];
            // alphaResults[(int)(alpha*20)-1][0] += totalAction[0];
            // alphaResults[(int)(alpha*20)-1][1] += totalAction[1];
            // alphaResults[(int)(alpha*20)-1][2] += totalAction[2];

        // cout << totalAction[0] << " " << totalAction[1] << " " << totalAction[2] << endl;
    }
}
   
    cout << "Results over 10 runs:\n";
    
    for(int i = 0; i < 10; i++) {
        float alpha = (i + 1) / 10.0;
        vector<int> results = alphaResults[i];
        cout << alpha << " " << results[0]/10 << " " << results[1]/10 << " " << results[2]/10 << endl;
    }

    // for(int i = 0; i < 26; i++) {
    //     float alpha = (i + 1) / 10.0;
    //     vector<int> results = alphaResults[i];
    //     cout << alpha << " " << results[0]/10 << " " << results[1]/10 << " " << results[2]/10 << endl;
    // }
   
    for (int i = 0; i < n; i++) {
        delete[] grid[i];
    }
    delete[] grid;
    
    return 0;
}