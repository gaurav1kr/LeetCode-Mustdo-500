
# Word Search II - Optimized C++ Solution

This is an optimized and concise C++ solution for [LeetCode 212: Word Search II](https://leetcode.com/problems/word-search-ii/) using a Trie (Prefix Tree) and backtracking. The Trie helps efficiently search for words, while the backtracking explores the board.

```cpp
class Solution {
public:
    struct TrieNode {
        TrieNode* children[26] = {nullptr};
        string word = "";
    };

    void insert(TrieNode* root, const string& word) {
        TrieNode* node = root;
        for (char c : word) {
            int index = c - 'a';
            if (!node->children[index]) 
                node->children[index] = new TrieNode();
            node = node->children[index];
        }
        node->word = word; // Store the word at the end node
    }

    void dfs(vector<vector<char>>& board, int i, int j, TrieNode* node, vector<string>& result) {
        char c = board[i][j];
        if (c == '#' || !node->children[c - 'a']) return;

        node = node->children[c - 'a'];
        if (!node->word.empty()) { // Found a word
            result.push_back(node->word);
            node->word = ""; // Avoid duplicates
        }

        board[i][j] = '#'; // Mark as visited
        int dirs[4][2] = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        for (auto& dir : dirs) {
            int x = i + dir[0], y = j + dir[1];
            if (x >= 0 && x < board.size() && y >= 0 && y < board[0].size()) {
                dfs(board, x, y, node, result);
            }
        }
        board[i][j] = c; // Restore the cell
    }

    vector<string> findWords(vector<vector<char>>& board, vector<string>& words) {
        TrieNode* root = new TrieNode();
        for (const string& word : words) {
            insert(root, word);
        }

        vector<string> result;
        for (int i = 0; i < board.size(); ++i) {
            for (int j = 0; j < board[0].size(); ++j) {
                dfs(board, i, j, root, result);
            }
        }

        return result;
    }
};
```

## Explanation

### 1. **Trie Construction**
- Build a Trie for all the words in the input list. Each node represents a letter, and the end of a word is marked by storing the word in the node.

### 2. **Backtracking**
- Start a DFS from each cell in the board.
- Check if the current letter exists in the Trie.
- Mark the cell as visited and explore its neighbors.
- If a word is found, add it to the result and mark it as used to avoid duplicates.

### 3. **Optimization**
- Use `Trie` to efficiently prune invalid paths.
- Avoid revisiting cells by marking them with `#` temporarily.

## Complexity

### Time Complexity
- **Trie Construction**: \(O(L \times W)\), where \(L\) is the average word length and \(W\) is the number of words.
- **Searching**: \(O(M \times N \times 4^K)\), where \(M \times N\) is the board size and \(K\) is the maximum word length.

### Space Complexity
- \(O(L \times W)\) for the Trie.

This solution efficiently combines the power of Trie and backtracking to solve the problem within a reasonable time for large inputs.
