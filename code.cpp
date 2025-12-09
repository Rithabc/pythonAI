
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
#define INT_MAX = 2 ^ 9;
#define FLT_MAX 3.402823422e+38F
#include <cstring>
#include <chrono>
#include <fstream>


namespace std
{
    template <>
    struct hash<std::pair<int, int>>
    {
        std::size_t operator()(const std::pair<int, int> &p) const noexcept
        {
            return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
        }
    };
}

using namespace std;

std::mt19937 gen(std::chrono::system_clock::now().time_since_epoch().count());

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

    dfs(ind, ind, n, grid);

    
    
    
    
    
    
    
    
    
    return grid;
}


int manhattan(int r1, int c1, int r2, int c2)
{
    return abs(r1 - r2) + abs(c1 - c2);
}










typedef pair<int, int> Pair;


typedef pair<double, pair<int, int>> pPair;


struct cell
{
    
    
    int parent_i, parent_j;
    
    double f, g, h;
};



bool isValid(int row, int col, int size)
{
    
    
    return (row >= 0) && (row < size) && (col >= 0) && (col < size);
}



bool isUnBlocked(int **grid, int row, int col)
{
    
    if (grid[row][col] == 0)
        return (true);
    else
        return (false);
}



bool isDestination(int row, int col, Pair dest)
{
    if (row == dest.first && col == dest.second)
        return (true);
    else
        return (false);
}


double calculateHValue(int row, int col, Pair dest)
{
    
    return ((double)sqrt(
        (row - dest.first) * (row - dest.first) + (col - dest.second) * (col - dest.second)));
}




vector<Pair> edit;
vector<string> directions;
void tracePath(vector<vector<cell>> cellDetails, Pair dest)
{
    
    int row = dest.first;
    int col = dest.second;

    stack<Pair> Path;

    while (!(cellDetails[row][col].parent_i == row && cellDetails[row][col].parent_j == col))
    {
        Path.push(make_pair(row, col));
        int temp_row = cellDetails[row][col].parent_i;
        int temp_col = cellDetails[row][col].parent_j;
        row = temp_row;
        col = temp_col;
    }

    int tr = row;
    int tc = col;

    Path.push(make_pair(row, col));
    while (!Path.empty())
    {
        pair<int, int> p = Path.top();
        Path.pop();
        
        
        if (p.first == tr - 1 && p.second == tc)
        {
            
            edit.push_back(make_pair(-1, 0));
            directions.push_back("UP");
        }
        else if (p.first == tr + 1 && p.second == tc)
        {
            
            edit.push_back(make_pair(1, 0));
            directions.push_back("DOWN");
        }
        else if (p.first == tr && p.second == tc - 1)
        {
            
            edit.push_back(make_pair(0, -1));
            directions.push_back("LEFT");
        }
        else if (p.first == tr && p.second == tc + 1)
        {
            
            edit.push_back(make_pair(0, 1));
            directions.push_back("RIGHT");
        }
        tr = p.first;
        tc = p.second;
        

        
    }

    return;
}
Pair tsrc, tdest;




void aStarSearch(int **grid, Pair src, Pair dest, int size)
{
    
    if (isValid(src.first, src.second, size) == false)
    {
        printf("Source is invalid\n");
        return;
    }

    
    if (isValid(dest.first, dest.second, size) == false)
    {
        printf("Destination is invalid\n");
        return;
    }

    
    if (isUnBlocked(grid, src.first, src.second) == false || isUnBlocked(grid, dest.first, dest.second) == false)
    {
        printf("Source or the destination is blocked\n");
        return;
    }

    
    if (isDestination(src.first, src.second, dest) == true)
    {
        
        return;
    }

    
    
    
    bool closedList[size][size];
    memset(closedList, false, sizeof(closedList));

    
    

    vector<vector<cell>> cellDetails(size, vector<cell>(size));

    int i, j;

    for (i = 0; i < size; i++)
    {
        for (j = 0; j < size; j++)
        {
            cellDetails[i][j].f = FLT_MAX;
            cellDetails[i][j].g = FLT_MAX;
            cellDetails[i][j].h = FLT_MAX;
            cellDetails[i][j].parent_i = -1;
            cellDetails[i][j].parent_j = -1;
        }
    }

    
    i = src.first, j = src.second;
    cellDetails[i][j].f = 0.0;
    cellDetails[i][j].g = 0.0;
    cellDetails[i][j].h = 0.0;
    cellDetails[i][j].parent_i = i;
    cellDetails[i][j].parent_j = j;

   
    set<pPair> openList;

    
    
    openList.insert(make_pair(0.0, make_pair(i, j)));

    
    
    bool foundDest = false;

    while (!openList.empty())
    {
        pPair p = *openList.begin();

        
        openList.erase(openList.begin());

        
        i = p.second.first;
        j = p.second.second;
        closedList[i][j] = true;

       

        
        double gNew, hNew, fNew;

        

        
        if (isValid(i - 1, j, size) == true)
        {
            
            
            if (isDestination(i - 1, j, dest) == true)
            {
                
                cellDetails[i - 1][j].parent_i = i;
                cellDetails[i - 1][j].parent_j = j;
                
                tracePath(cellDetails, dest);
                foundDest = true;
                return;
            }
            
            
            
            else if (closedList[i - 1][j] == false && isUnBlocked(grid, i - 1, j) == true)
            {
                gNew = cellDetails[i][j].g + 1.0;
                hNew = calculateHValue(i - 1, j, dest);
                fNew = gNew + hNew;

                
                
                
                
                
                
                
                
                if (cellDetails[i - 1][j].f == FLT_MAX || cellDetails[i - 1][j].f > fNew)
                {
                    openList.insert(make_pair(
                        fNew, make_pair(i - 1, j)));

                    
                    cellDetails[i - 1][j].f = fNew;
                    cellDetails[i - 1][j].g = gNew;
                    cellDetails[i - 1][j].h = hNew;
                    cellDetails[i - 1][j].parent_i = i;
                    cellDetails[i - 1][j].parent_j = j;
                }
            }
        }

        

        
        if (isValid(i + 1, j, size) == true)
        {
            
            
            if (isDestination(i + 1, j, dest) == true)
            {
                
                cellDetails[i + 1][j].parent_i = i;
                cellDetails[i + 1][j].parent_j = j;
                
                tracePath(cellDetails, dest);
                foundDest = true;
                return;
            }
            
            
            
            else if (closedList[i + 1][j] == false && isUnBlocked(grid, i + 1, j) == true)
            {
                gNew = cellDetails[i][j].g + 1.0;
                hNew = calculateHValue(i + 1, j, dest);
                fNew = gNew + hNew;

                
                
                
                
                
                
                
                
                if (cellDetails[i + 1][j].f == FLT_MAX || cellDetails[i + 1][j].f > fNew)
                {
                    openList.insert(make_pair(
                        fNew, make_pair(i + 1, j)));
                    
                    cellDetails[i + 1][j].f = fNew;
                    cellDetails[i + 1][j].g = gNew;
                    cellDetails[i + 1][j].h = hNew;
                    cellDetails[i + 1][j].parent_i = i;
                    cellDetails[i + 1][j].parent_j = j;
                }
            }
        }

        

        
        if (isValid(i, j + 1, size) == true)
        {
            
            
            if (isDestination(i, j + 1, dest) == true)
            {
                
                cellDetails[i][j + 1].parent_i = i;
                cellDetails[i][j + 1].parent_j = j;
                
                tracePath(cellDetails, dest);
                foundDest = true;
                return;
            }

            
            
            
            else if (closedList[i][j + 1] == false && isUnBlocked(grid, i, j + 1) == true)
            {
                gNew = cellDetails[i][j].g + 1.0;
                hNew = calculateHValue(i, j + 1, dest);
                fNew = gNew + hNew;

                
                
                
                
                
                
                
                
                if (cellDetails[i][j + 1].f == FLT_MAX || cellDetails[i][j + 1].f > fNew)
                {
                    openList.insert(make_pair(
                        fNew, make_pair(i, j + 1)));

                    
                    cellDetails[i][j + 1].f = fNew;
                    cellDetails[i][j + 1].g = gNew;
                    cellDetails[i][j + 1].h = hNew;
                    cellDetails[i][j + 1].parent_i = i;
                    cellDetails[i][j + 1].parent_j = j;
                }
            }
        }

        

        
        if (isValid(i, j - 1, size) == true)
        {
            
            
            if (isDestination(i, j - 1, dest) == true)
            {
                
                cellDetails[i][j - 1].parent_i = i;
                cellDetails[i][j - 1].parent_j = j;
                
                tracePath(cellDetails, dest);
                foundDest = true;
                return;
            }

            
            
            
            else if (closedList[i][j - 1] == false && isUnBlocked(grid, i, j - 1) == true)
            {
                gNew = cellDetails[i][j].g + 1.0;
                hNew = calculateHValue(i, j - 1, dest);
                fNew = gNew + hNew;

                
                
                
                
                
                
                
                
                if (cellDetails[i][j - 1].f == FLT_MAX || cellDetails[i][j - 1].f > fNew)
                {
                    openList.insert(make_pair(
                        fNew, make_pair(i, j - 1)));

                    
                    cellDetails[i][j - 1].f = fNew;
                    cellDetails[i][j - 1].g = gNew;
                    cellDetails[i][j - 1].h = hNew;
                    cellDetails[i][j - 1].parent_i = i;
                    cellDetails[i][j - 1].parent_j = j;
                }
            }
        }
    }

    
    
    
    
    
    if (foundDest == false)
        printf("Failed to find the Destination Cell\n");

    return;
}

int minVal = 9999999;

int algoUtil(int size, int **grid, Pair src, Pair dest, vector<pair<int, int>> bots, queue<Pair> temp ,bool random = true)
{
    tsrc = src;
    tdest = dest;
    directions.clear();
    while (!temp.empty())
    {
        
        if(random){
        int num = generateRandom(0, bots.size() - 1);
        aStarSearch(grid, bots[num] , dest, size);
        }else{
        aStarSearch(grid, temp.front(), dest, size);
        }
        if (directions.size() > minVal)
        {
            directions.clear();
            edit.clear();
            return 9999999;
        }
        

        for (int i = 0; i < bots.size(); i++)
        {
            for (int j = 0; j < edit.size(); j++)
            {
                if (bots[i].first == dest.first && bots[i].second == dest.second)
                {
                }
                else
                {
                    if (bots[i].first + edit[j].first < 0 || bots[i].first + edit[j].first >= size || bots[i].second + edit[j].second < 0 || bots[i].second + edit[j].second >= size)
                        continue;

                    bots[i].first += edit[j].first;
                    bots[i].second += edit[j].second;
                    if (grid[bots[i].first][bots[i].second] == 1)
                    {
                        bots[i].first -= edit[j].first;
                        bots[i].second -= edit[j].second;
                    }
                }
                
                

                
                
                

            }
        }
        
        temp = queue<Pair>();
        for (int i = 0; i < bots.size(); i++)
        {
            if (bots[i].first == dest.first && bots[i].second == dest.second)
            {
            }
            else
            {
                
                temp.push(bots[i]);
            }
        }
        edit.clear();

        
        
        
    }
    
    
    
    
    
    
    
    if (directions.size() < minVal)
        minVal = directions.size();
    return directions.size();
}

struct hash_pair
{
   
    Pair p;
    vector<string> v;
    int len;
    hash_pair(Pair p, vector<string> v, int len) : p(p), v(v), len(len) {};
};



void efficient(int size, vector<Pair> bots, vector<string> &efficientAns, vector<vector<int>> &efficientVis, int **grid, unordered_map<pair<int, int>, int> &efficientMap)
{
    
    stack<Pair> st;
    st.push(bots[0]);
    int prevI = bots[0].first;
    int prevJ = bots[0].second;
    while (!efficientMap.size() == 0)
    {
        Pair t = st.top();
        
        efficientMap.erase(t);
        int i = t.first;
        int j = t.second;
        efficientVis[i][j] = 1;

        if (prevI == i - 1)
            efficientAns.push_back("DOWN");
        if (prevI == i + 1)
            efficientAns.push_back("UP");
        if (prevJ == j - 1)
            efficientAns.push_back("RIGHT");
        if (prevJ == j + 1)
            efficientAns.push_back("LEFT");

        prevI = i;
        prevJ = j;

        if (i - 1 >= 0 && efficientVis[i - 1][j] == 0 && grid[i - 1][j] == 0)
        {
            st.push(make_pair(i - 1, j));
        }
        else if (i + 1 < size && efficientVis[i + 1][j] == 0 && grid[i + 1][j] == 0)
        {
            st.push(make_pair(i + 1, j));
        }
        else if (j - 1 >= 0 && efficientVis[i][j - 1] == 0 && grid[i][j - 1] == 0)
        {
            st.push(make_pair(i, j - 1));
        }
        else if (j + 1 < size && efficientVis[i][j + 1] == 0 && grid[i][j + 1] == 0)
        {
            st.push(make_pair(i, j + 1));
        }
        else
        {
            
            
            st.pop();
        }

    }
    tdest = make_pair(prevI, prevJ);
}




int main()
{
    ofstream trainingFile("training_data.csv");
    trainingFile << "grid_size,num_obstacles,obstacle_density,num_bots,optimal_moves\n";
    
    ofstream gridFile("grid_maps.txt");

    cout << "Enter the max: ";
    int n;
    cin >> n;
    

    for (int k = 2; k <= n; k = k + 1)
    {
        
        float efficientAvg = 0;
        float optimalAvg = 0;
        float avg = 0;

        float efficientTime = 0;
        float optimalTime = 0;
        float avgTime = 0;
        
        for (int w = 0; w < 200; w++)
        {
            vector<vector<int>> generatedGrid = generateGrid(k);

            int** grid = new int*[k];
            for (int i = 0; i < k; i++)
            {
                grid[i] = new int[k];
                for (int j = 0; j < k; j++)
                {
                    
                    grid[i][j] = generatedGrid[i][j];
                }
            }

            
            
            
            
            
            
            
            



            vector<Pair> bots;
            unordered_map<pair<int, int>, int> efficientMap;
            queue<Pair> temp;
            for (int i = 0; i < k; i++)
            {
                for (int j = 0; j < k; j++)
                {
                    if (grid[i][j] == 0)
                    {
                        bots.push_back(make_pair(i, j));
                        efficientMap[make_pair(i, j)]++;
                        temp.push(make_pair(i, j));
                    }
                }
            }
            auto start = std::chrono::high_resolution_clock::now();

            avg += algoUtil(k, grid, bots[0], bots[generateRandom(0, bots.size() - 1)], bots, temp);
            minVal = 9999999;

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            
            
            avgTime += duration.count() / 1000000.0;
            
            
            
            

            auto oStart = std::chrono::high_resolution_clock::now();
            int max = 999999;
            vector<string> optimalAns;
            Pair dest = make_pair(-1, -1);
            for (int a = bots.size() -1; a >= 0; a--)
            {
                
                
                   
                        int optimal = algoUtil(k, grid, bots[0], bots[a], bots, temp,false);
                        if (optimal < max)
                        {
                            minVal = optimal;
                            max = optimal;
                            optimalAns = directions;
                            dest = bots[a];
                        }
                    
                    
            }
        
            
            
            minVal = 9999999;
            auto oEnd = std::chrono::high_resolution_clock::now();
            auto oDuration = std::chrono::duration_cast<std::chrono::microseconds>(oEnd - oStart);
            optimalTime += oDuration.count() / 1000000.0;
            optimalAvg += optimalAns.size();

            int numObstacles = 0;
            for (int i = 0; i < k; i++)
                for (int j = 0; j < k; j++)
                    if (grid[i][j] == 1) numObstacles++;
            
            float obstacleDensity = (float)numObstacles / (k * k);
            
            trainingFile << k << "," << numObstacles << "," << obstacleDensity << "," 
                        << bots.size() << "," << optimalAns.size() << "\n";
            
            gridFile << k << "\n";
            for (int i = 0; i < k; i++) {
                for (int j = 0; j < k; j++)
                    gridFile << grid[i][j];
                gridFile << "\n";
            }
            gridFile << optimalAns.size() << "\n---\n";

            vector<string> efficientAns;
            vector<vector<int>> efficientVis(k, vector<int>(k, 0));
            auto eStart = std::chrono::high_resolution_clock::now();
            efficient(k, bots, efficientAns, efficientVis, grid, efficientMap);
            auto eEnd = std::chrono::high_resolution_clock::now();
            auto eDuration = std::chrono::duration_cast<std::chrono::microseconds>(eEnd - eStart);
            efficientTime += eDuration.count() / 1000000.0;
            efficientAvg += efficientAns.size();
        }
        
        cout <<k << " " << avg / 3 << " " << efficientAvg / 3 << " " << optimalAvg / 3 << " " << avgTime / 3 << " " << efficientTime / 3 << " " << optimalTime / 3 << "\n";
    }

    trainingFile.close();
    gridFile.close();
    cout << "Training data saved to training_data.csv and grid_maps.txt\n";

    return (0);
}