# Solution to LeetCode Problem: Unique Paths II

## Problem Description
The problem can be found at [Unique Paths II - LeetCode](https://leetcode.com/problems/unique-paths-ii/description/).

You are given an `m x n` grid filled with non-negative integers representing a grid of obstacles:
- `0` represents an empty cell.
- `1` represents an obstacle.

Your task is to determine the number of unique paths from the top-left corner to the bottom-right corner, assuming that you can only move down or right at any step.

If the starting point or ending point is blocked (contains `1`), the number of unique paths is `0`.

---

## Optimized C++ Solution
This approach uses **dynamic programming (DP)** with **space optimization**.

```cpp
#include <vector>
using namespace std;

class Solution 
{
public:
    int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) 
    {
        int m = obstacleGrid.size();
        int n = obstacleGrid[0].size();
        
        // If the starting point or ending point is blocked, no paths are possible
        if (obstacleGrid[0][0] == 1 || obstacleGrid[m-1][n-1] == 1) {
            return 0;
        }
        
        // DP array to store the number of ways to reach each cell in the current row
        vector<int> dp(n, 0);
        dp[0] = 1; // Start point
        
        for (int i = 0; i < m; ++i) 
	{
            for (int j = 0; j < n; ++j) 
	    {
                if (obstacleGrid[i][j] == 1) 
		{
                    dp[j] = 0; // No path through an obstacle
                } 
		else if (j > 0) 
		{
                    dp[j] += dp[j - 1]; // Add paths from the left cell
                }
            }
        }
        
        return dp[n - 1];
    }
};
```

---

## Key Points
1. **Space Optimization**: Instead of a 2D DP array (`dp[m][n]`), this solution uses a 1D array (`dp[n]`) to store the current row's state.
2. **Time Complexity**: `O(m × n)` as we iterate through the entire grid once.
3. **Edge Cases**: 
    - If the starting point or ending point is blocked (`obstacleGrid[0][0] == 1` or `obstacleGrid[m-1][n-1] == 1`), return `0`.
    - Handle grids with only one row or one column appropriately.

---

## Explanation
1. Initialize a 1D DP array `dp` of size `n` (number of columns).
    - Set `dp[0] = 1` since there is exactly one way to start at the top-left corner.
2. Iterate through each cell in the grid row by row.
    - If the cell is an obstacle (`obstacleGrid[i][j] == 1`), set `dp[j] = 0`.
    - Otherwise, update `dp[j]` as `dp[j] += dp[j-1]` to account for paths coming from the left.
3. Return `dp[n-1]`, which contains the number of unique paths to the bottom-right corner.

---

## Example
### Input:
```cpp
obstacleGrid = {
    {0, 0, 0},
    {0, 1, 0},
    {0, 0, 0}
};
```
### Output:
```cpp
Solution().uniquePathsWithObstacles(obstacleGrid); // Returns 2
```

### Explanation:
- There are two paths to reach the bottom-right corner (ignoring the obstacle).

---

## Complexity Analysis
- **Time Complexity**: `O(m × n)`
    - We iterate through every cell in the grid once.
- **Space Complexity**: `O(n)`
    - Only a single row's worth of data is stored at a time in the `dp` array.

---

This solution is efficient and minimizes memory usage, making it suitable for large grids.
