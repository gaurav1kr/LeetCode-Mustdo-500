
# Minimum Path Sum (LeetCode)

This document provides an optimized C++ solution for the LeetCode problem ["Minimum Path Sum"](https://leetcode.com/problems/minimum-path-sum/description/).

---

## Explanation

This solution uses dynamic programming with space optimization. Instead of maintaining a full 2D `dp` table, we use a single row (`dp` array) to store the minimum path sums for the current row.

### Approach:
1. Use a 1D vector `dp` to store the current row's minimum path sums.
2. Traverse the grid and update the `dp` array in-place to compute the minimum path sum for each cell.
3. This reduces the space complexity from **O(m × n)** to **O(n)**, where `n` is the number of columns.

---

## Optimized C++ Solution

```cpp
#include <vector>
#include <algorithm>
using namespace std;

class Solution {
public:
    int minPathSum(vector<vector<int>>& grid) {
        int m = grid.size();      // Number of rows
        int n = grid[0].size();   // Number of columns
        vector<int> dp(n, 0);

        // Initialize the dp array with the first row
        dp[0] = grid[0][0];
        for (int j = 1; j < n; ++j) {
            dp[j] = dp[j - 1] + grid[0][j];
        }

        // Iterate through the rest of the rows
        for (int i = 1; i < m; ++i) {
            dp[0] += grid[i][0]; // Update the first column
            for (int j = 1; j < n; ++j) {
                dp[j] = min(dp[j], dp[j - 1]) + grid[i][j];
            }
        }

        return dp[n - 1];
    }
};
```

---

## Complexity Analysis

- **Time Complexity**: **O(m × n)**  
  We visit each cell in the grid exactly once.

- **Space Complexity**: **O(n)**  
  The solution uses a single 1D array of size `n` to store the current row's minimum path sums.

---

## Example Usage

```cpp
int main() {
    Solution solution;
    vector<vector<int>> grid = {
        {1, 3, 1},
        {1, 5, 1},
        {4, 2, 1}
    };
    int result = solution.minPathSum(grid);
    // Output: 7
    return 0;
}
```

### Input/Output Example:
Input:
```text
grid = [[1,3,1],[1,5,1],[4,2,1]]
```

Output:
```text
7
```

---

## Notes
This optimized solution is particularly useful for large grids where memory usage needs to be minimized.

