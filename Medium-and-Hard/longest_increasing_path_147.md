
# Longest Increasing Path in a Matrix (LeetCode)

## Problem Description

Given an `m x n` integer matrix, return the length of the longest increasing path in the matrix.

From each cell, you can either move in four directions: left, right, up, or down. You may not move diagonally or move outside the boundary (i.e., wrap-around is not allowed).

Link to the problem: [LeetCode - Longest Increasing Path in a Matrix](https://leetcode.com/problems/longest-increasing-path-in-a-matrix/description/)

---

## Optimized C++ Solution

This solution uses **Depth-First Search (DFS)** with **memoization** to avoid redundant calculations:

```cpp
class Solution {
public:
    int longestIncreasingPath(vector<vector<int>>& matrix) {
        int rows = matrix.size(), cols = matrix[0].size();
        vector<vector<int>> memo(rows, vector<int>(cols, -1));
        int maxLength = 0;

        function<int(int, int)> dfs = [&](int r, int c) -> int {
            if (memo[r][c] != -1) return memo[r][c];
            int maxPath = 1;
            vector<pair<int, int>> directions = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
            for (auto [dr, dc] : directions) {
                int nr = r + dr, nc = c + dc;
                if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && matrix[nr][nc] > matrix[r][c]) {
                    maxPath = max(maxPath, 1 + dfs(nr, nc));
                }
            }
            return memo[r][c] = maxPath;
        };

        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                maxLength = max(maxLength, dfs(r, c));
            }
        }

        return maxLength;
    }
};
```

---

## Explanation

### 1. **DFS with Memoization**
- Each cell is visited at most once due to the memoization array (`memo`).
- Memoization stores the longest increasing path starting from each cell, reducing redundant computations.

### 2. **Helper Function (`dfs`)**
- The recursive function computes the longest increasing path starting from the current cell.
- It explores all 4 possible directions (up, down, left, right) and considers only valid moves where the next cell value is greater than the current cell value.

### 3. **Outer Loop**
- The main loop ensures all cells are considered as possible starting points for the longest path.

---

## Complexity Analysis

### Time Complexity
- **O(m × n)**: Each cell is processed once during the DFS and memoized.

### Space Complexity
- **O(m × n)**: For the memoization table.
- **O(m × n)**: For the implicit recursion stack in the worst case.

---

## Key Points
- Efficient solution leveraging memoization to reduce redundant calculations.
- Handles edge cases where matrix dimensions are minimal or all values are equal.

---

Let me know if you need further clarification or improvements!
