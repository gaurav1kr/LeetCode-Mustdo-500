## Optimized C++ Solution for Word Search (LeetCode)

### **Approach**
- Use **DFS (Depth-First Search)** to explore the grid.
- Apply **backtracking** to mark visited cells and revert them after exploring.
- **Pruning** is applied by checking if the first letter exists in the grid before proceeding.
- **Time Complexity**: **O(N * 3^L)**, where **N** is the number of cells and **L** is the word length.
- **Space Complexity**: **O(L)** due to recursion depth.

---

### **C++ Optimized Solution**
```cpp
class Solution {
public:
    int rows, cols;
    
    bool dfs(vector<vector<char>>& board, string& word, int i, int j, int index) {
        // Base Case: If we matched all characters
        if (index == word.length()) return true;
        
        // Boundary Conditions & Mismatch Check
        if (i < 0 || j < 0 || i >= rows || j >= cols || board[i][j] != word[index]) 
            return false;
        
        // Mark cell as visited by temporarily changing its value
        char temp = board[i][j];
        board[i][j] = '#';  // Mark as visited
        
        // Explore all four possible directions
        bool found = dfs(board, word, i + 1, j, index + 1) ||
                     dfs(board, word, i - 1, j, index + 1) ||
                     dfs(board, word, i, j + 1, index + 1) ||
                     dfs(board, word, i, j - 1, index + 1);
        
        // Restore original character
        board[i][j] = temp;
        
        return found;
    }
    
    bool exist(vector<vector<char>>& board, string word) {
        rows = board.size();
        cols = board[0].size();
        
        // Pruning: Check if all characters in 'word' exist in the board
        unordered_map<char, int> boardCount, wordCount;
        for (auto& row : board)
            for (char c : row) boardCount[c]++;
        for (char c : word) wordCount[c]++;
        
        for (auto& [c, count] : wordCount)
            if (boardCount[c] < count) return false;
        
        // Start DFS from every cell
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (dfs(board, word, i, j, 0)) return true;
            }
        }
        return false;
    }
};
```

---

### **Why This is Optimized**
1. **Backtracking**: We revert visited cells to their original state.
2. **Pruning**: Before running DFS, we check if the board has enough characters of each type.
3. **Efficient DFS**: We avoid unnecessary computations by stopping early when a mismatch is found.

This solution works efficiently for large grids while keeping memory usage low. 🚀
