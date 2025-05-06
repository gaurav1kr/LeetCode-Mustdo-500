##1 ****[Problem Link]https://leetcode.com/problems/number-of-islands/****
```cpp
class Solution {
public:
    void dfs(vector<vector<char>>& grid, int i, int j) {
        if (i < 0 || j < 0 || i >= grid.size() || j >= grid[0].size() || grid[i][j] == '0') return;
        grid[i][j] = '0';
        dfs(grid, i+1, j);
        dfs(grid, i-1, j);
        dfs(grid, i, j+1);
        dfs(grid, i, j-1);
    }

    int numIslands(vector<vector<char>>& grid) {
        int count = 0;
        for (int i = 0; i < grid.size(); i++) {
            for (int j = 0; j < grid[0].size(); j++) {
                if (grid[i][j] == '1') {
                    dfs(grid, i, j);
                    count++;
                }
            }
        }
        return count;
    }
};
```

##2 ****[Problem Link]https://leetcode.com/problems/word-search/****
```cpp
class Solution {
    bool dfs(vector<vector<char>>& board, string& word, int i, int j, int index) {
        if (index == word.size()) return true;
        if (i < 0 || j < 0 || i >= board.size() || j >= board[0].size() || board[i][j] != word[index]) return false;

        char temp = board[i][j];
        board[i][j] = '#';
        bool found = dfs(board, word, i+1, j, index+1) ||
                     dfs(board, word, i-1, j, index+1) ||
                     dfs(board, word, i, j+1, index+1) ||
                     dfs(board, word, i, j-1, index+1);
        board[i][j] = temp;
        return found;
    }

public:
    bool exist(vector<vector<char>>& board, string word) {
        for (int i = 0; i < board.size(); i++) {
            for (int j = 0; j < board[0].size(); j++) {
                if (dfs(board, word, i, j, 0)) return true;
            }
        }
        return false;
    }
};
```

##3 ****[Problem Link]https://leetcode.com/problems/rotate-image/****
```cpp
class Solution {
public:
    void rotate(vector<vector<int>>& matrix) {
        int n = matrix.size();
        for (int i = 0; i < n; ++i)
            for (int j = i; j < n; ++j)
                swap(matrix[i][j], matrix[j][i]);
        for (int i = 0; i < n; ++i)
            reverse(matrix[i].begin(), matrix[i].end());
    }
};
```

##4 ****[Problem Link]https://leetcode.com/problems/maximal-square/****
```cpp
class Solution {
public:
    int maximalSquare(vector<vector<char>>& matrix) {
        int m = matrix.size(), n = matrix[0].size(), maxLen = 0;
        vector<vector<int>> dp(m+1, vector<int>(n+1, 0));
        for (int i = 1; i <= m; ++i) {
            for (int j = 1; j <= n; ++j) {
                if (matrix[i-1][j-1] == '1') {
                    dp[i][j] = min({dp[i-1][j], dp[i][j-1], dp[i-1][j-1]}) + 1;
                    maxLen = max(maxLen, dp[i][j]);
                }
            }
        }
        return maxLen * maxLen;
    }
};
```

##5 ****[Problem Link]https://leetcode.com/problems/minimum-path-sum/****
```cpp
class Solution {
public:
    int minPathSum(vector<vector<int>>& grid) {
        int m = grid.size(), n = grid[0].size();
        for (int i = 1; i < m; ++i) grid[i][0] += grid[i-1][0];
        for (int j = 1; j < n; ++j) grid[0][j] += grid[0][j-1];
        for (int i = 1; i < m; ++i)
            for (int j = 1; j < n; ++j)
                grid[i][j] += min(grid[i-1][j], grid[i][j-1]);
        return grid[m-1][n-1];
    }
};
```