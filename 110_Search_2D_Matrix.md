
# Solution to LeetCode Problem: Search a 2D Matrix

This is an optimized C++ solution for the LeetCode problem ["Search a 2D Matrix"](https://leetcode.com/problems/search-a-2d-matrix/). The solution uses a binary search approach by treating the matrix as a flattened 1D array.

## Code
```cpp
#include <vector>
using namespace std;

class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        if (matrix.empty() || matrix[0].empty()) return false;

        int rows = matrix.size();
        int cols = matrix[0].size();
        int left = 0, right = rows * cols - 1;

        while (left <= right) {
            int mid = left + (right - left) / 2;
            int midValue = matrix[mid / cols][mid % cols]; // Convert 1D index to 2D

            if (midValue == target) {
                return true;
            } else if (midValue < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }

        return false;
    }
};
```

## Explanation

### 1. Flatten the Matrix Conceptually
- Treat the 2D matrix as a 1D array.
- Use the following formulas to convert between 1D and 2D indices:
  - **Row**: `mid / cols`
  - **Column**: `mid % cols`

### 2. Binary Search
- Initialize `left = 0` and `right = rows * cols - 1`.
- Perform binary search:
  - Calculate the middle index `mid`.
  - Access the value at the middle index using `matrix[mid / cols][mid % cols]`.
  - Compare `midValue` with the `target`:
    - If they are equal, return `true`.
    - If `midValue < target`, search the right half.
    - Otherwise, search the left half.

### 3. Edge Cases
- If the matrix is empty or its first row is empty, return `false`.

## Complexity
- **Time Complexity**: O(log(rows * cols)), as we are performing binary search on the matrix.
- **Space Complexity**: O(1), as no extra space is used.

## Example
### Input
```cpp
vector<vector<int>> matrix = {
    {1, 3, 5, 7},
    {10, 11, 16, 20},
    {23, 30, 34, 60}
};
int target = 3;

Solution solution;
bool result = solution.searchMatrix(matrix, target);
```

### Output
```cpp
// Output: true
```

## Notes
This solution is efficient and adheres to the constraints of the problem. By treating the matrix as a flattened 1D array, the implementation avoids the complexity of working with row and column boundaries directly, simplifying the logic.
