# Cherry Pickup Problem Solution

## Problem Overview

The Cherry Pickup problem involves two robots starting at the top-left corner of a grid and moving towards the bottom-right corner while collecting the maximum number of cherries. The grid cells can contain cherries (positive values) or obstacles (denoted by `-1`).

## Solution Explanation

### Key Concepts

1. **Grid Representation**:
   - The grid is represented as a 2D vector.
   - Cells can either contain cherries or obstacles.

2. **Dynamic Programming Table**:
   - A 3D DP table `dp[r1][c1][r2]` is used to store the maximum cherries collected for specific robot positions:
     - Robot 1 (`r1`, `c1`)
     - Robot 2 (`r2`, `c2`), where `c2` is derived from the total moves made.

3. **Recursion**:
   - The `solve` function explores all possible paths for both robots.
   - Each robot can move either down or right.
   - The function checks for boundaries and obstacles.
   - When both robots are in the same cell, cherries are only counted once.

4. **Base Cases**:
   - If either robot goes out of bounds or encounters an obstacle, return `INT_MIN`.
   - If Robot 1 reaches the bottom-right corner, collect cherries from that cell.

5. **Memoization**:
   - The DP table is filled only when necessary, and previously computed values are reused.

### Code Implementation

```cpp
class Solution 
{
public:
    int solve(vector<vector<int>>& grid, int r1, int c1, int r2, int c2, int n, vector<vector<vector<int>>>& dp)
    {
        if (r1 >= n || r2 >= n || c1 >= n || c2 >= n || grid[r1][c1] == -1 || grid[r2][c2] == -1)
        {
            return INT_MIN;
        }

        if (r1 == n - 1 && c1 == n - 1)
        {
            return grid[r1][c1];
        }

        if (dp[r1][c1][r2] != -1) 
        {
            return dp[r1][c1][r2];
        }
       
        int cherries = (r1 == r2 && c1 == c2) ? grid[r1][c1] : grid[r1][c1] + grid[r2][c2];

        int f1 = solve(grid, r1, c1 + 1, r2, c2 + 1, n, dp); // both move right
        int f2 = solve(grid, r1 + 1, c1, r2, c2 + 1, n, dp); // Robot 1 down, Robot 2 right
        int f3 = solve(grid, r1 + 1, c1, r2 + 1, c2, n, dp); // both down
        int f4 = solve(grid, r1, c1 + 1, r2 + 1, c2, n, dp); // Robot 1 right, Robot 2 down

        cherries += max({f1, f2, f3, f4});
        return dp[r1][c1][r2] = cherries; // Store result in dp table
    }

    int cherryPickup(vector<vector<int>>& grid) 
    {
        int n = grid.size();
        vector<vector<vector<int>>> dp(n, vector<vector<int>>(n, vector<int>(n, -1)));
        int ans = solve(grid, 0, 0, 0, 0, n, dp);
        return max(0, ans); // Return max between 0 and computed cherries
    }
};

