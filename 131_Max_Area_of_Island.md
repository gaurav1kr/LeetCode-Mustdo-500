
# Max Area of Island (LeetCode Problem)

## Problem Description
You can find the full problem description [here](https://leetcode.com/problems/max-area-of-island/).

## Optimized C++ Solution

```cpp
#include <vector>
#include <algorithm>
using namespace std;

class Solution {
public:
    int maxAreaOfIsland(vector<vector<int>>& grid) {
        int maxArea = 0;
        int rows = grid.size(), cols = grid[0].size();

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                if (grid[i][j] == 1) {
                    maxArea = max(maxArea, dfs(grid, i, j));
                }
            }
        }

        return maxArea;
    }

private:
    int dfs(vector<vector<int>>& grid, int i, int j) {
        // Boundary checks
        if (i < 0 || i >= grid.size() || j < 0 || j >= grid[0].size() || grid[i][j] == 0) {
            return 0;
        }

        // Mark the current cell as visited
        grid[i][j] = 0;

        // Explore in all 4 directions
        return 1 + dfs(grid, i + 1, j) + dfs(grid, i - 1, j) + dfs(grid, i, j + 1) + dfs(grid, i, j - 1);
    }
};
```

## Explanation

1. **DFS Traversal**:
   - When a cell with value `1` is found, we start a DFS to calculate the area of the connected island.
   - Each cell of the island is marked as `0` (visited) to avoid revisiting.

2. **Recursive DFS**:
   - The `dfs` function checks boundaries and ensures we don't process already visited or water cells.
   - Recursively calculates the area of the island.

3. **Efficiency**:
   - **Time Complexity**: \(O(m 	imes n)\), where \(m\) and \(n\) are the grid dimensions. Each cell is visited once.
   - **Space Complexity**: \(O(m 	imes n)\) in the worst case for the recursion stack, in case of a completely filled grid.

This implementation is clean and optimized for readability and performance.
