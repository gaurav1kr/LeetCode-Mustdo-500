
# Optimized C++ Solution for N-Queens Problem

## Problem Link
[LeetCode - N-Queens](https://leetcode.com/problems/n-queens)

## Solution
```cpp
class Solution {
public:
    vector<vector<string>> solveNQueens(int n) {
        vector<vector<string>> results;
        vector<string> board(n, string(n, '.'));
        vector<bool> columns(n, false), diag1(2 * n - 1, false), diag2(2 * n - 1, false);
        solve(0, n, board, results, columns, diag1, diag2);
        return results;
    }

private:
    void solve(int row, int n, vector<string>& board, vector<vector<string>>& results, 
               vector<bool>& columns, vector<bool>& diag1, vector<bool>& diag2) {
        if (row == n) {
            results.push_back(board);
            return;
        }

        for (int col = 0; col < n; ++col) {
            int d1 = row - col + n - 1;
            int d2 = row + col;
            if (columns[col] || diag1[d1] || diag2[d2]) continue;

            board[row][col] = 'Q';
            columns[col] = diag1[d1] = diag2[d2] = true;

            solve(row + 1, n, board, results, columns, diag1, diag2);

            board[row][col] = '.';
            columns[col] = diag1[d1] = diag2[d2] = false;
        }
    }
};
```

## Explanation

### Backtracking
- We attempt to place queens row by row.
- If a queen can be placed at a position without conflict, we proceed to the next row.
- Otherwise, we backtrack by removing the queen and trying the next position.

### Conflict Checking
- `columns[col]`: Tracks if a queen exists in the column.
- `diag1[row - col + n - 1]`: Tracks if a queen exists in the left-to-right diagonal.
- `diag2[row + col]`: Tracks if a queen exists in the right-to-left diagonal.

### Optimization
- Avoids repeatedly scanning the board by using arrays to track conflicts.

## Complexity
- **Time Complexity**: O(N!) in the worst case as we attempt to place queens row by row.
- **Space Complexity**: O(N^2) for the board and additional arrays to track conflicts.
