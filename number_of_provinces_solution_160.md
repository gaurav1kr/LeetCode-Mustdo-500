# Number of Provinces Solution

## Problem Description
The problem "Number of Provinces" is to determine the number of connected components (provinces) in an undirected graph represented by an adjacency matrix.

Link: [LeetCode - Number of Provinces](https://leetcode.com/problems/number-of-provinces/)

## Approach
We solve this problem using Depth-First Search (DFS) to explore all connected cities starting from an unvisited city.

### Key Concepts
1. **Graph Representation**: The input is represented as an adjacency matrix `isConnected`, where `isConnected[i][j] = 1` indicates a direct connection between city `i` and city `j`.
2. **Provinces**: Each connected component in the graph corresponds to one province.
3. **DFS**: Used to traverse the graph and mark all cities in the same province as visited.

## C++ Solution
```cpp
#include <vector>
using namespace std;

class Solution {
public:
    void dfs(int node, vector<vector<int>>& isConnected, vector<bool>& visited) {
        visited[node] = true;
        for (int neighbor = 0; neighbor < isConnected.size(); ++neighbor) {
            if (isConnected[node][neighbor] == 1 && !visited[neighbor]) {
                dfs(neighbor, isConnected, visited);
            }
        }
    }

    int findCircleNum(vector<vector<int>>& isConnected) {
        int n = isConnected.size();
        vector<bool> visited(n, false);
        int provinces = 0;

        for (int i = 0; i < n; ++i) {
            if (!visited[i]) {
                ++provinces;
                dfs(i, isConnected, visited);
            }
        }

        return provinces;
    }
};
```

## Explanation
1. **DFS Function**:
   - Marks the current city as visited.
   - Recursively visits all directly connected unvisited cities.
2. **Visited Vector**:
   - Tracks whether a city has been visited to avoid redundant work.
3. **Province Count**:
   - For each unvisited city, increment the province count and call DFS to explore its connected component.

## Complexity Analysis
- **Time Complexity**: 
  - The adjacency matrix is of size `n × n`, and we iterate through it during DFS. Hence, the time complexity is **O(n^2)**.
- **Space Complexity**: 
  - The `visited` vector takes **O(n)** space.

## Example
### Input
```plaintext
isConnected = [
    [1, 1, 0],
    [1, 1, 0],
    [0, 0, 1]
]
```
### Output
```plaintext
2
```
### Explanation
- Cities 0 and 1 are connected, forming one province.
- City 2 is not connected to any other city, forming a second province.

## Conclusion
This solution efficiently determines the number of provinces using a graph traversal technique and works well for small to medium-sized graphs.
