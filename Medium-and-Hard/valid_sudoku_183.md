# Valid Sudoku Solution

This document provides an optimized C++ solution for the ["Valid Sudoku"](https://leetcode.com/problems/valid-sudoku/) problem on LeetCode.

## Problem Description
A Sudoku board (9x9) must satisfy the following rules:

1. Each row must contain the digits `1-9` without repetition.
2. Each column must contain the digits `1-9` without repetition.
3. Each of the 9 sub-boxes of the grid must contain the digits `1-9` without repetition.

Empty cells are represented by `'.'` and can be ignored when validating the board.

## Optimized C++ Solution
```cpp
#include <vector>
#include <unordered_set>

using namespace std;

class Solution {
public:
    bool isValidSudoku(vector<vector<char>>& board) {
        // Arrays of hash sets to track rows, columns, and sub-boxes
        unordered_set<char> rows[9];
        unordered_set<char> cols[9];
        unordered_set<char> boxes[9];

        // Iterate through the board
        for (int i = 0; i < 9; ++i) {
            for (int j = 0; j < 9; ++j) {
                char num = board[i][j];
                if (num == '.') continue; // Skip empty cells

                // Calculate the index for the 3x3 sub-box
                int boxIndex = (i / 3) * 3 + j / 3;

                // Check for duplicates in the row, column, or box
                if (rows[i].count(num) || cols[j].count(num) || boxes[boxIndex].count(num)) {
                    return false;
                }

                // Add the number to the respective row, column, and box sets
                rows[i].insert(num);
                cols[j].insert(num);
                boxes[boxIndex].insert(num);
            }
        }

        return true; // All checks passed
    }
};
```

## Explanation

### Data Structures
- **`unordered_set`**: Used to track digits in each row, column, and 3x3 sub-box efficiently.
  - `rows[9]`: Tracks digits in each row.
  - `cols[9]`: Tracks digits in each column.
  - `boxes[9]`: Tracks digits in each 3x3 sub-box.

### Algorithm
1. Traverse the entire 9x9 board.
2. For each cell:
   - If the cell is empty (`'.'`), skip it.
   - Compute the index of the corresponding 3x3 sub-box using `(i / 3) * 3 + j / 3`.
   - Check if the current digit exists in the respective row, column, or sub-box. If it does, return `false`.
   - Otherwise, add the digit to the corresponding row, column, and sub-box sets.
3. If all cells are validated without conflicts, return `true`.

### Complexity Analysis
- **Time Complexity**: `O(81) = O(1)`
  - The board size is fixed (9x9), so we perform constant work for each cell.
- **Space Complexity**: `O(81) = O(1)`
  - In the worst case, all cells are filled, and each digit is stored in a set.

## Notes
- This solution is efficient and adheres to the problem constraints.
- It leverages hash sets for constant-time checks and insertions.

## References
- Problem Link: [Valid Sudoku](https://leetcode.com/problems/valid-sudoku/)
