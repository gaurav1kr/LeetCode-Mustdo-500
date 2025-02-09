# Optimized C++ Solution for LeetCode Problem: Surrounded Regions

Here is an optimized and concise C++ solution for the "Surrounded Regions" problem using Depth-First Search (DFS):

```cpp
#include <vector>
using namespace std;

class Solution {
public:
    void solve(vector<vector<char>>& board) {
        int rows = board.size(), cols = board[0].size();
        
        // Lambda function for DFS traversal
        auto dfs = [&](int r, int c, auto& dfs) -> void {
            if (r < 0 || r >= rows || c < 0 || c >= cols || board[r][c] != 'O') return;
            board[r][c] = 'T'; // Mark the cell as temporarily safe
            dfs(r + 1, c, dfs);
            dfs(r - 1, c, dfs);
            dfs(r, c + 1, dfs);
            dfs(r, c - 1, dfs);
        };

        // Run DFS on 'O's connected to the border
        for (int r = 0; r < rows; ++r) {
            dfs(r, 0, dfs);
            dfs(r, cols - 1, dfs);
        }
        for (int c = 0; c < cols; ++c) {
            dfs(0, c, dfs);
            dfs(rows - 1, c, dfs);
        }

        // Convert 'O' to 'X' (surrounded regions) and 'T' back to 'O' (safe regions)
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                if (board[r][c] == 'O') board[r][c] = 'X';
                else if (board[r][c] == 'T') board[r][c] = 'O';
            }
        }
    }
};
```

## Explanation:
1. **Mark Border-Connected 'O':**
   - We traverse the border and mark all 'O's connected to the border with a temporary marker 'T'.
   - This ensures these regions will not be flipped to 'X'.

2. **Flip and Restore:**
   - Traverse the board:
     - Change remaining 'O' (surrounded) to 'X'.
     - Restore 'T' (temporarily marked) back to 'O'.

3. **Lambda DFS:**
   - The `dfs` function is implemented as a lambda for simplicity. It recursively marks connected 'O's as 'T'.

## Complexity:
- **Time Complexity:** \(O(m \times n)\), where \(m\) and \(n\) are the dimensions of the board.
- **Space Complexity:** \(O(m \times n)\) in the worst case due to the recursion stack.

This code is concise, readable, and meets performance requirements.
