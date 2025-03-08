
# LeetCode Problem 240: Search a 2D Matrix II

This problem requires searching for a target value in an \(m 	imes n\) matrix sorted in ascending order both row-wise and column-wise. Below is the optimized C++ solution.

## Optimized Solution (Time Complexity: \(O(m + n)\)):

```cpp
#include <vector>
using namespace std;

class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        if (matrix.empty() || matrix[0].empty()) return false;

        int rows = matrix.size();
        int cols = matrix[0].size();

        // Start from the top-right corner
        int row = 0;
        int col = cols - 1;

        while (row < rows && col >= 0) {
            if (matrix[row][col] == target) {
                return true; // Found the target
            } else if (matrix[row][col] > target) {
                col--; // Move left
            } else {
                row++; // Move down
            }
        }

        return false; // Target not found
    }
};
```

## Explanation

1. **Start at the top-right corner**:
   - The reason for starting here is that:
     - Moving left reduces the column value (and the numbers in that direction are smaller).
     - Moving down increases the row value (and the numbers in that direction are larger).

2. **Iterate through the matrix**:
   - Compare the current element `matrix[row][col]` with the target.
   - If the element is greater than the target, move left (decrement column).
   - If the element is smaller than the target, move down (increment row).

3. **Stop conditions**:
   - Stop if the row index exceeds the matrix height (`row >= rows`).
   - Stop if the column index becomes negative (`col < 0`).

## Complexity Analysis

- **Time Complexity**: \(O(m + n)\), where \(m\) is the number of rows and \(n\) is the number of columns. This is because, in the worst case, you traverse either the entire first row or the entire last column.
- **Space Complexity**: \(O(1)\), as no extra space is used.
