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

using namespace std;

typedef pair<int, int> Pair;
typedef pair<double, pair<int, int>> pPair;

// Hash function for Pair
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

        // Check all 4 directions
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
    float p_beep = exp(-alpha * (dist - 1)); // formula: e^{-α(dist-1)}

    return beep ? p_beep : (1.0 - p_beep);
}


// Simulate beep: 1 = found (same cell), 0 = beep, -1 = no beep
int simulateBeep(Pair locator, Pair trueRoombaPos, float alpha) {
    if (locator.first == trueRoombaPos.first && 
        locator.second == trueRoombaPos.second) {
        return 1; // Found!
    }
  
    
    int dist = manhattan(locator.first, locator.second, 
                        trueRoombaPos.first, trueRoombaPos.second);
    float p_beep = exp(-alpha * (dist - 1));
    
    float r = (float)rand() / RAND_MAX;
    return (r < p_beep) ? 0 : -1;
}

// Bayesian update
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

// Get cell with maximum probability
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

// Update belief after a movement command
void updateBeliefAfterMove(unordered_map<Pair, float>& belief, 
                          string move, int** grid, int size) {
    unordered_map<Pair, float> newBelief;
    
    for (const auto& entry : belief) {
        Pair currentPos = entry.first;
        float prob = entry.second;
        
        // Calculate where this position would move to
        int newI = currentPos.first;
        int newJ = currentPos.second;
        
        if (move == "UP") newI--;
        else if (move == "DOWN") newI++;
        else if (move == "LEFT") newJ--;
        else if (move == "RIGHT") newJ++;
        
        // Check if move is valid
        if (isValid(newI, newJ, size) && isUnBlocked(grid, newI, newJ)) {
            // Bot moves to new position
            Pair newPos = make_pair(newI, newJ);
            newBelief[newPos] += prob;
        } else {
            // Bot stays in current position (blocked)
            newBelief[currentPos] += prob;
        }
    }
    
    belief = newBelief;
}

void printGrid(const vector<vector<int>>& grid, Pair locator, Pair maxProbCell) {
    int n = grid.size();
    //cout << "\nGrid (L=Locator, M=MaxProb, X=Wall):\n";
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == locator.first && j == locator.second) {
                //cout << " L ";
            } else if (i == maxProbCell.first && j == maxProbCell.second) {
                //cout << " M ";
            } else if (grid[i][j] == 1) {
                //cout << " X ";
            } else {
                //cout << " . ";
            }
        }
        //cout << endl;
    }
}

void printBeliefMatrix(const unordered_map<Pair, float>& belief, 
                       const vector<vector<int>>& gridVec, 
                       Pair locator) {
    int n = gridVec.size();
    //cout << "\nBelief Matrix (L=Locator, X=Wall):\n";
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == locator.first && j == locator.second) {
                //cout << "  L   ";
            } else if (gridVec[i][j] == 1) {
                //cout << "  X   ";
            } else {
                auto it = belief.find(make_pair(i, j));
                if (it != belief.end() && it->second > 0.0001) {
                    printf("%.3f ", it->second);
                } else {
                    //cout << " ---  ";
                }
            }
        }
        //cout << endl;
    }
    //cout << endl;
}

// Baseline Strategy 2
int baselineStrategy2(int **grid, int size, Pair locatorStart, 
                      Pair trueRoombaPos, float alpha, 
                      vector<Pair> allOpenCells,
                      const vector<vector<int>>& gridVec) {
    
    // Initialize uniform belief
    unordered_map<Pair, float> belief;
    for (auto cell : allOpenCells) {
        belief[cell] = 1.0 / allOpenCells.size();
    }
    
    Pair locator = locatorStart;
    Pair roombaPos = trueRoombaPos; // Track actual roomba position
    int totalActions = 0;
    bool roombaFound = false;
    
    //cout << "\n=== BASELINE STRATEGY 2 ===\n";
    //cout << "True Roomba Position (hidden): (" << roombaPos.first 
         << ", " << roombaPos.second << ")\n";
    //cout << "Locator Start: (" << locator.first << ", " << locator.second << ")\n";
    
    //cout << "\n--- Initial Belief ---";
    printBeliefMatrix(belief, gridVec, locator);
    
    while (!roombaFound && totalActions < 10000) { // Safety limit
        // Step 1: Run detector (sense action)
        int beepResult = simulateBeep(locator, roombaPos, alpha);
        totalActions++;
        
        //cout << "\n========================================\n";
        //cout << "Action " << totalActions << ": SENSE at (" 
             << locator.first << ", " << locator.second << ") - ";
        
        if (beepResult == 1) {
            //cout << "FOUND ROOMBA!\n";
            roombaFound = true;
            break;
        }
        
        bool beepHeard = (beepResult == 0);
        //cout << (beepHeard ? "BEEP" : "NO BEEP") << "\n";
        
        // Step 2: Update probabilities
        updateBelief(belief, locator, beepHeard, alpha, false);
        
        //cout << "\n--- After Sensing ---";
        printBeliefMatrix(belief, gridVec, locator);
        
        // Step 3: Get cell with maximal probability
        Pair maxProbCell = getMaxProbabilityCell(belief);
        float maxProb = belief[maxProbCell];
        //cout << "Max probability cell: (" << maxProbCell.first << ", " 
             << maxProbCell.second << ") with prob=" << maxProb << "\n";
        
        // Step 4: Calculate path from maxProbCell to locator
        directions.clear();
        edit.clear();
        aStarSearch(grid, maxProbCell, locator, size);
        
        // If already at destination, just continue sensing
        if (directions.empty()) {
            //cout << "Max prob cell is at locator - continuing\n";
            continue;
        }
        
        // Step 5: Issue ONE command to move bot at maxProbCell closer to locator
        string nextMove = directions[0];
        totalActions++;
        
        //cout << "\nAction " << totalActions << ": MOVE " << nextMove << "\n";
        
        // Update belief distribution after move
        updateBeliefAfterMove(belief, nextMove, grid, size);
        
        //cout << "\n--- After Move ---";
        printBeliefMatrix(belief, gridVec, locator);
        
        // Update true roomba position (it also follows the command)
        int newI = roombaPos.first;
        int newJ = roombaPos.second;
        if (nextMove == "UP") newI--;
        else if (nextMove == "DOWN") newI++;
        else if (nextMove == "LEFT") newJ--;
        else if (nextMove == "RIGHT") newJ++;
        
        if (isValid(newI, newJ, size) && isUnBlocked(grid, newI, newJ)) {
            roombaPos.first = newI;
            roombaPos.second = newJ;
        }
        
        //cout << "Roomba now at: (" << roombaPos.first 
             << ", " << roombaPos.second << ")\n";
    }
    
    //cout << "\n========================================\n";
    //cout << "=== RESULT ===\n";
    //cout << "Total Actions (Sense + Move): " << totalActions << "\n";
    //cout << "Roomba found at: (" << roombaPos.first 
         << ", " << roombaPos.second << ")\n\n";
    
    return totalActions;
}

// Baseline Strategy 2 - FIXED
// Original logic: Move Roomba at maxProbCell toward locator
// Fix: Add termination condition when maxProbCell == locator
int baselineStrategy2_fixed(int **grid, int size, Pair locatorStart, 
                            Pair trueRoombaPos, float alpha, 
                            vector<Pair> allOpenCells,
                            const vector<vector<int>>& gridVec) {
    
    // Initialize uniform belief
    unordered_map<Pair, float> belief;
    for (auto cell : allOpenCells) {
        belief[cell] = 1.0 / allOpenCells.size();
    }
    
    Pair locator = locatorStart;
    Pair roombaPos = trueRoombaPos; // Track actual roomba position
    int totalActions = 0;
    bool roombaFound = false;
    
    //cout << "\n=== BASELINE STRATEGY 2 (FIXED) ===\n";
    //cout << "True Roomba Position (hidden): (" << roombaPos.first 
         << ", " << roombaPos.second << ")\n";
    //cout << "Locator Start: (" << locator.first << ", " << locator.second << ")\n";
    
    //cout << "\n--- Initial Belief ---";
    printBeliefMatrix(belief, gridVec, locator);
    
    while (!roombaFound && totalActions < 10000) { // Safety limit
        // Step 1: Run detector (sense action)
        int beepResult = simulateBeep(locator, roombaPos, alpha);
        totalActions++;
        
        //cout << "\n========================================\n";
        //cout << "Action " << totalActions << ": SENSE at (" 
             << locator.first << ", " << locator.second << ") - ";
        
        if (beepResult == 1) {
            //cout << "FOUND ROOMBA!\n";
            roombaFound = true;
            break;
        }
        
        bool beepHeard = (beepResult == 0);
        //cout << (beepHeard ? "BEEP" : "NO BEEP") << "\n";
        
        // Step 2: Update probabilities
        updateBelief(belief, locator, beepHeard, alpha, false);
        
        //cout << "\n--- After Sensing ---";
        printBeliefMatrix(belief, gridVec, locator);
        
        // Step 3: Get cell with maximal probability
        Pair maxProbCell = getMaxProbabilityCell(belief);
        float maxProb = belief[maxProbCell];
        //cout << "Max probability cell: (" << maxProbCell.first << ", " 
             << maxProbCell.second << ") with prob=" << maxProb << "\n";
        
        // Step 4: Calculate path from maxProbCell to locator
        directions.clear();
        edit.clear();
        aStarSearch(grid, maxProbCell, locator, size);
        
        // If already at destination, just continue sensing
        if (directions.empty()) {
            //cout << "Max prob cell is at locator - continuing\n";
            
            // FIX: Check if Roomba is actually at locator
            if (roombaPos.first == locator.first && roombaPos.second == locator.second) {
                // Will be found on next sense
                continue;
            }
            
            // If maxProb is at locator but Roomba isn't actually there,
            // belief will update on next sense and we continue
            continue;
        }
        
        // Step 5: Issue ONE command to move bot at maxProbCell closer to locator
        string nextMove = directions[0];
        totalActions++;
        
        //cout << "\nAction " << totalActions << ": MOVE " << nextMove << "\n";
        
        // Update belief distribution after move
        updateBeliefAfterMove(belief, nextMove, grid, size);
        
        //cout << "\n--- After Move ---";
        printBeliefMatrix(belief, gridVec, locator);
        
        // Update true roomba position (it also follows the command)
        int newI = roombaPos.first;
        int newJ = roombaPos.second;
        if (nextMove == "UP") newI--;
        else if (nextMove == "DOWN") newI++;
        else if (nextMove == "LEFT") newJ--;
        else if (nextMove == "RIGHT") newJ++;
        
        if (isValid(newI, newJ, size) && isUnBlocked(grid, newI, newJ)) {
            roombaPos.first = newI;
            roombaPos.second = newJ;
        }
        
        //cout << "Roomba now at: (" << roombaPos.first 
             << ", " << roombaPos.second << ")\n";
    }
    
    //cout << "\n========================================\n";
    //cout << "=== RESULT ===\n";
    //cout << "Total Actions (Sense + Move): " << totalActions << "\n";
    //cout << "Roomba found at: (" << roombaPos.first 
         << ", " << roombaPos.second << ")\n\n";
    
    return totalActions;
}

vector<Pair> locatorPath; // Keep track of path
// Baseline Strategy 2 - TRUE OPTIMIZATION
// Move the LOCATOR toward the max probability cell
int baselineStrategy2_trueOptimized(int **grid, int size, Pair locatorStart, 
                                     Pair trueRoombaPos, float alpha, 
                                     vector<Pair> allOpenCells,
                                     const vector<vector<int>>& gridVec) {

    // Initialize uniform belief
    unordered_map<Pair, float> belief;
    for (auto cell : allOpenCells) {
        belief[cell] = 1.0 / allOpenCells.size();
    }

    Pair locator = locatorStart;
    Pair roombaPos = trueRoombaPos; // Track actual Roomba position (stays still)
    int totalActions = 0;
    bool roombaFound = false;

    //cout << "\n=== BASELINE STRATEGY 2 - TRUE OPTIMIZED ===\n";
    //cout << "True Roomba Position (hidden): (" << roombaPos.first 
         << ", " << roombaPos.second << ")\n";
    //cout << "Locator Start: (" << locator.first << ", " << locator.second << ")\n";

    while (!roombaFound && totalActions < 10000) { // Safety limit
        // Step 1: Sense at current locator position
        int beepResult = simulateBeep(locator, roombaPos, alpha);
        totalActions++;

        //cout << "\n========================================\n";
        //cout << "Action " << totalActions << ": SENSE at (" 
             << locator.first << ", " << locator.second << ") - ";

        if (beepResult == 1) {
            //cout << "FOUND ROOMBA!\n";
            roombaFound = true;
            break;
        }

        bool beepHeard = (beepResult == 0);
        //cout << (beepHeard ? "BEEP" : "NO BEEP") << "\n";

        // Step 2: Update belief after sensing
        updateBelief(belief, locator, beepHeard, alpha, false);

        // Print belief matrix
        //cout << "\n--- Belief After Sensing ---";
        printBeliefMatrix(belief, gridVec, locator);

        // Step 3: Get max probability cell
        Pair maxProbCell = getMaxProbabilityCell(belief);
        float maxProb = belief[maxProbCell];
        //cout << "Max probability cell: (" << maxProbCell.first << ", " 
             << maxProbCell.second << ") with prob=" << maxProb << "\n";

        // Step 4: Check if already at max probability cell
        if (locator.first == maxProbCell.first && locator.second == maxProbCell.second) {
            //cout << "Locator already at max prob cell - continuing sensing\n";
            continue;
        }

        // Step 5: Compute path from locator to maxProbCell
        directions.clear();
        edit.clear();
        aStarSearch(grid, locator, maxProbCell, size);

        if (directions.empty()) {
            //cout << "No path found - continuing\n";
            continue;
        }

        // Step 6: Move LOCATOR one step toward max probability cell
        string nextMove = directions[0];
        totalActions++;

        //cout << "\nAction " << totalActions << ": MOVE LOCATOR " << nextMove << "\n";

        // Update locator position
        int newI = locator.first;
        int newJ = locator.second;
        
        if (nextMove == "UP") newI--;
        else if (nextMove == "DOWN") newI++;
        else if (nextMove == "LEFT") newJ--;
        else if (nextMove == "RIGHT") newJ++;

        // Check if move is valid
        if (isValid(newI, newJ, size) && isUnBlocked(grid, newI, newJ)) {
            locator.first = newI;
            locator.second = newJ;
            //cout << "Locator moved to: (" << locator.first 
                 << ", " << locator.second << ")\n";
        } else {
            //cout << "Move blocked - locator stays at: (" << locator.first 
                 << ", " << locator.second << ")\n";
        }

        // Roomba stays in place (it's not moving)
        // No belief update needed since we're just repositioning the sensor
        
        //cout << "Roomba remains at: (" << roombaPos.first 
             << ", " << roombaPos.second << ")\n";
    }

    //cout << "\n========================================\n";
    //cout << "=== RESULT ===\n";
    //cout << "Total Actions (Sense + Move): " << totalActions << "\n";
    //cout << "Roomba found at: (" << roombaPos.first << ", " << roombaPos.second << ")\n";
    //cout << "Locator ended at: (" << locator.first << ", " << locator.second << ")\n\n";

    return totalActions;
}

int baselineStrategy2_pathOnly(int **grid, int size, Pair locatorStart, 
                      Pair trueRoombaPos, float alpha, 
                      vector<Pair> allOpenCells) {
    
    unordered_map<Pair, float> belief;
    for (auto cell : allOpenCells) {
        belief[cell] = 1.0 / allOpenCells.size();
    }
    
    Pair locator = locatorStart;
    Pair roombaPos = trueRoombaPos;
    bool roombaFound = false;

    while (!roombaFound && locatorPath.size() < 10000) { // safety limit
        // Step 1: sense
        int beepResult = simulateBeep(locator, roombaPos, alpha);
        if (beepResult == 1) { // found
            roombaFound = true;
            locatorPath.push_back(locator);
            break;
        }
        bool beepHeard = (beepResult == 0);

        // Step 2: update belief
        updateBelief(belief, locator, beepHeard, alpha, false);

        // Step 3: move to max probability cell
        Pair maxProbCell = getMaxProbabilityCell(belief);
        directions.clear();
        edit.clear();
        aStarSearch(grid, locator, maxProbCell, size); // move locator towards maxProbCell

        if (!directions.empty()) {
            string move = directions[0]; // take only 1 step
            locatorPath.push_back(locator); // record current position

            // move locator
            if (move == "UP") locator.first--;
            else if (move == "DOWN") locator.first++;
            else if (move == "LEFT") locator.second--;
            else if (move == "RIGHT") locator.second++;

            // update belief after move
            updateBeliefAfterMove(belief, move, grid, size);
        }
    }

    // print the path
    //cout << "\nLocator Path to Roomba:\n";
    for (auto p : locatorPath) {
        //cout << "(" << p.first << "," << p.second << ") ";
    }
    //cout << "(" << locator.first << "," << locator.second << ") "; // final cell
    //cout << "\nTotal steps: " << locatorPath.size() << endl;

    return locatorPath.size();
}

// Baseline Strategy 2 optimized with belief printing
int baselineStrategy2_optimized(int **grid, int size, Pair locatorStart, 
                                Pair trueRoombaPos, float alpha, 
                                vector<Pair> allOpenCells,
                                const vector<vector<int>>& gridVec) {

    // Initialize uniform belief
    unordered_map<Pair, float> belief;
    for (auto cell : allOpenCells) {
        belief[cell] = 1.0 / allOpenCells.size();
    }

    Pair locator = locatorStart;
    Pair roombaPos = trueRoombaPos; // Track actual Roomba position
    int totalActions = 0;
    bool roombaFound = false;

    //cout << "\n=== BASELINE STRATEGY 2 OPTIMIZED ===\n";
    //cout << "True Roomba Position (hidden): (" << roombaPos.first 
         << ", " << roombaPos.second << ")\n";
    //cout << "Locator Start: (" << locator.first << ", " << locator.second << ")\n";

    while (!roombaFound && totalActions < 10000) { // Safety limit
        // Step 1: Sense
        int beepResult = simulateBeep(locator, roombaPos, alpha);
        totalActions++;

        //cout << "\n========================================\n";
        //cout << "Action " << totalActions << ": SENSE at (" 
             << locator.first << ", " << locator.second << ") - ";

        if (beepResult == 1) {
            //cout << "FOUND ROOMBA!\n";
            roombaFound = true;
            break;
        }

        bool beepHeard = (beepResult == 0);
        //cout << (beepHeard ? "BEEP" : "NO BEEP") << "\n";

        // Update belief after sensing
        updateBelief(belief, locator, beepHeard, alpha, false);

        // Print belief matrix
        //cout << "\n--- Belief After Sensing ---";
        printBeliefMatrix(belief, gridVec, locator);

        // Step 2: Get max probability cell
        Pair maxProbCell = getMaxProbabilityCell(belief);
        float maxProb = belief[maxProbCell];
        //cout << "Max probability cell: (" << maxProbCell.first << ", " 
             << maxProbCell.second << ") with prob=" << maxProb << "\n";

        // Step 3: Compute path from maxProbCell to locator
        directions.clear();
        edit.clear();
        aStarSearch(grid, maxProbCell, locator, size);

        if (directions.empty()) {
            // Already at max probability cell
            //cout << "Max prob cell is at locator - continuing sensing\n";
            continue;
        }

        // Step 4: Move bot one step toward locator
        string nextMove = directions[0];
        totalActions++;

        //cout << "\nAction " << totalActions << ": MOVE " << nextMove << "\n";

        // Convert move string to delta coordinates
        int di = 0, dj = 0;
        if (nextMove == "UP") { di = -1; dj = 0; }
        else if (nextMove == "DOWN") { di = 1; dj = 0; }
        else if (nextMove == "LEFT") { di = 0; dj = -1; }
        else if (nextMove == "RIGHT") { di = 0; dj = 1; }

        // Locator stays in place (not moving)
        locator.first += 0;
        locator.second += 0;

        // Update belief after move
        updateBeliefAfterMove(belief, nextMove, grid, size);

        // Print belief matrix after move
        //cout << "\n--- Belief After Move ---";
        printBeliefMatrix(belief, gridVec, locator);

        // Update actual Roomba position
        int newI = roombaPos.first + di;
        int newJ = roombaPos.second + dj;
        if (isValid(newI, newJ, size) && isUnBlocked(grid, newI, newJ)) {
            roombaPos.first = newI;
            roombaPos.second = newJ;
        }

        //cout << "Roomba now at: (" << roombaPos.first 
             << ", " << roombaPos.second << ")\n";
    }

    //cout << "\nTotal Actions (Sense + Move): " << totalActions << "\n";
    //cout << "Roomba found at: (" << roombaPos.first << ", " << roombaPos.second << ")\n\n";

    return totalActions;
}

// Baseline Strategy 2 - TRUE OPTIMIZATION
// Move the LOCATOR toward the max probability cell
int baselineStrategy2_trueOptimized2(int **grid, int size, Pair locatorStart, 
                                     Pair trueRoombaPos, float alpha, 
                                     vector<Pair> allOpenCells,
                                     const vector<vector<int>>& gridVec) {

    // Initialize uniform belief
    unordered_map<Pair, float> belief;
    for (auto cell : allOpenCells) {
        belief[cell] = 1.0 / allOpenCells.size();
    }

    Pair locator = locatorStart;
    Pair roombaPos = trueRoombaPos; // Track actual Roomba position (stays still)
    int totalActions = 0;
    bool roombaFound = false;

    //cout << "\n=== BASELINE STRATEGY 2 - TRUE OPTIMIZED ===\n";
    //cout << "True Roomba Position (hidden): (" << roombaPos.first 
         << ", " << roombaPos.second << ")\n";
    //cout << "Locator Start: (" << locator.first << ", " << locator.second << ")\n";

    while (!roombaFound && totalActions < 10000) { // Safety limit
        // Step 1: Sense at current locator position
        int beepResult = simulateBeep(locator, roombaPos, alpha);
        totalActions++;

        //cout << "\n========================================\n";
        //cout << "Action " << totalActions << ": SENSE at (" 
             << locator.first << ", " << locator.second << ") - ";

        if (beepResult == 1) {
            //cout << "FOUND ROOMBA!\n";
            roombaFound = true;
            break;
        }

        bool beepHeard = (beepResult == 0);
        //cout << (beepHeard ? "BEEP" : "NO BEEP") << "\n";

        // Step 2: Update belief after sensing
        updateBelief(belief, locator, beepHeard, alpha, false);

        // Print belief matrix
        //cout << "\n--- Belief After Sensing ---";
        printBeliefMatrix(belief, gridVec, locator);

        // Step 3: Get max probability cell
        Pair maxProbCell = getMaxProbabilityCell(belief);
        float maxProb = belief[maxProbCell];
        //cout << "Max probability cell: (" << maxProbCell.first << ", " 
             << maxProbCell.second << ") with prob=" << maxProb << "\n";

        // Step 4: Check if already at max probability cell
        if (locator.first == maxProbCell.first && locator.second == maxProbCell.second) {
            //cout << "Locator already at max prob cell - continuing sensing\n";
            continue;
        }

        // Step 5: Compute path from locator to maxProbCell
        directions.clear();
        edit.clear();
        aStarSearch(grid, locator, maxProbCell, size);

        if (directions.empty()) {
            //cout << "No path found - continuing\n";
            continue;
        }

        // Step 6: Move LOCATOR one step toward max probability cell
        string nextMove = directions[0];
        totalActions++;

        //cout << "\nAction " << totalActions << ": MOVE LOCATOR " << nextMove << "\n";

        // Update locator position
        int newI = locator.first;
        int newJ = locator.second;
        
        if (nextMove == "UP") newI--;
        else if (nextMove == "DOWN") newI++;
        else if (nextMove == "LEFT") newJ--;
        else if (nextMove == "RIGHT") newJ++;

        // Check if move is valid
        if (isValid(newI, newJ, size) && isUnBlocked(grid, newI, newJ)) {
            locator.first = newI;
            locator.second = newJ;
            //cout << "Locator moved to: (" << locator.first 
                 << ", " << locator.second << ")\n";
        } else {
            //cout << "Move blocked - locator stays at: (" << locator.first 
                 << ", " << locator.second << ")\n";
        }

        // Roomba stays in place (it's not moving)
        //cout << "Roomba remains at: (" << roombaPos.first 
             << ", " << roombaPos.second << ")\n";
        
        // The loop will continue and sense again from the new locator position
    }

    //cout << "\n========================================\n";
    //cout << "=== RESULT ===\n";
    //cout << "Total Actions (Sense + Move): " << totalActions << "\n";
    //cout << "Roomba found at: (" << roombaPos.first << ", " << roombaPos.second << ")\n";
    //cout << "Locator ended at: (" << locator.first << ", " << locator.second << ")\n\n";

    return totalActions;
}

// Baseline Strategy 2 - TRUE OPTIMIZATION (FIXED)
// Move the LOCATOR toward the max probability cell
// Roomba stays stationary - NO belief propagation after moves
int baselineStrategy2_trueOptimizedFixed(int **grid, int size, Pair locatorStart, 
                                          Pair trueRoombaPos, float alpha, 
                                          vector<Pair> allOpenCells,
                                          const vector<vector<int>>& gridVec) {

    // Initialize uniform belief
    unordered_map<Pair, float> belief;
    for (auto cell : allOpenCells) {
        belief[cell] = 1.0 / allOpenCells.size();
    }

    Pair locator = locatorStart;
    Pair roombaPos = trueRoombaPos; // Roomba position (stays still!)
    int totalActions = 0;
    bool roombaFound = false;

    //cout << "\n=== BASELINE STRATEGY 2 - TRUE OPTIMIZED (FIXED) ===\n";
    //cout << "True Roomba Position (hidden): (" << roombaPos.first 
         << ", " << roombaPos.second << ")\n";
    //cout << "Locator Start: (" << locator.first << ", " << locator.second << ")\n";

    while (!roombaFound && totalActions < 10000) { // Safety limit
        // Step 1: Sense at current locator position
        int beepResult = simulateBeep(locator, roombaPos, alpha);
        totalActions++;

        //cout << "\n========================================\n";
        //cout << "Action " << totalActions << ": SENSE at (" 
             << locator.first << ", " << locator.second << ") - ";

        if (beepResult == 1) {
            //cout << "FOUND ROOMBA!\n";
            roombaFound = true;
            break;
        }

        bool beepHeard = (beepResult == 0);
        //cout << (beepHeard ? "BEEP" : "NO BEEP") << "\n";

        // Step 2: Update belief after sensing
        updateBelief(belief, locator, beepHeard, alpha, false);

        // Print belief matrix
        //cout << "\n--- Belief After Sensing ---";
        printBeliefMatrix(belief, gridVec, locator);

        // Step 3: Get max probability cell
        Pair maxProbCell = getMaxProbabilityCell(belief);
        float maxProb = belief[maxProbCell];
        //cout << "Max probability cell: (" << maxProbCell.first << ", " 
             << maxProbCell.second << ") with prob=" << maxProb << "\n";

        // Step 4: Check if already at max probability cell
        if (locator.first == maxProbCell.first && locator.second == maxProbCell.second) {
            //cout << "Locator already at max prob cell - continuing sensing\n";
            continue;
        }

        // Step 5: Compute path from locator to maxProbCell
        directions.clear();
        edit.clear();
        aStarSearch(grid, locator, maxProbCell, size);

        if (directions.empty()) {
            //cout << "No path found - continuing\n";
            continue;
        }

        // Step 6: Move LOCATOR one step toward max probability cell
        string nextMove = directions[0];
        totalActions++;

        //cout << "\nAction " << totalActions << ": MOVE LOCATOR " << nextMove << "\n";

        // Update locator position
        int newI = locator.first;
        int newJ = locator.second;
        
        if (nextMove == "UP") newI--;
        else if (nextMove == "DOWN") newI++;
        else if (nextMove == "LEFT") newJ--;
        else if (nextMove == "RIGHT") newJ++;

        // Check if move is valid
        if (isValid(newI, newJ, size) && isUnBlocked(grid, newI, newJ)) {
            locator.first = newI;
            locator.second = newJ;
            //cout << "Locator moved to: (" << locator.first 
                 << ", " << locator.second << ")\n";
        } else {
            //cout << "Move blocked - locator stays at: (" << locator.first 
                 << ", " << locator.second << ")\n";
        }

        // CRITICAL: Roomba stays in place, so NO updateBeliefAfterMove!
        // The belief about where the Roomba is doesn't change just because we moved
        
        //cout << "Roomba remains at: (" << roombaPos.first 
             << ", " << roombaPos.second << ")\n";
        
        // Next iteration will sense from new locator position and update belief
    }

    //cout << "\n========================================\n";
    //cout << "=== RESULT ===\n";
    //cout << "Total Actions (Sense + Move): " << totalActions << "\n";
    //cout << "Roomba found at: (" << roombaPos.first << ", " << roombaPos.second << ")\n";
    //cout << "Locator ended at: (" << locator.first << ", " << locator.second << ")\n\n";

    return totalActions;
}


int main() {
    srand(time(0));
    
    // 5x5 grid
    vector<vector<int>> gridVec = {
        {0, 0, 1, 0, 0},
        {0, 0, 1, 0, 0},
        {0, 0, 1, 0, 0},
        {0, 1, 1, 1, 0},
        {0, 0, 0, 0, 0}
    };
    
    int n = gridVec.size();
    
    // Convert to int**
    int** grid = new int*[n];
    for (int i = 0; i < n; i++) {
        grid[i] = new int[n];
        for (int j = 0; j < n; j++) {
            grid[i][j] = gridVec[i][j];
        }
    }
    
    // Collect all open cells
    vector<Pair> allOpenCells;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (grid[i][j] == 0) {
                allOpenCells.push_back(make_pair(i, j));
            }
        }
    }
    
    // Set up scenario
    Pair locatorStart = make_pair(0, 0);
    Pair trueRoombaPos = make_pair(4, 4);
    float alpha = 0.5;
    
    // Run baseline strategy 2
    int totalActions = baselineStrategy2_fixed(grid, n, locatorStart, 
                                         trueRoombaPos, alpha, allOpenCells, gridVec);

    // int totalSteps = baselineStrategy2_pathOnly(grid, n, locatorStart, trueRoombaPos, alpha, allOpenCells);
    int totalAction  = baselineStrategy2_trueOptimizedFixed(grid, n, locatorStart, 
        trueRoombaPos, alpha, 
        allOpenCells, gridVec);
        
        // int totalActions2 = baselineStrategy2_optimized(grid, n, locatorStart, 
        //     trueRoombaPos, alpha, 
        //     allOpenCells, gridVec);
        //     //cout << endl;
            //cout << endl;
            //cout << endl;
            //cout << endl;
            //cout << "\nTotal Actions in Baseline Strategy 2: " << totalActions << endl;
            //cout << "\nTotal Actions in Baseline Strategy 2 TRUE OPTIMIZED: " << totalAction << endl;
    // //cout << "\nTotal Actions in Baseline Strategy 2 OPTIMIZED: " << totalActions2 << endl;
    
    // Cleanup
    for (int i = 0; i < n; i++) {
        delete[] grid[i];
    }
    delete[] grid;
    
    return 0;
}