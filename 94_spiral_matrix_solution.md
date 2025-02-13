# Spiral Matrix Solution in C++

## Problem Description
Given an \(m \times n\) matrix, return all elements of the matrix in spiral order. This involves traversing the matrix in a clockwise spiral pattern, starting from the top-left corner.

## Optimized and Concise C++ Code
```cpp
#include <vector>
using namespace std;

class Solution {
public:
    vector<int> spiralOrder(vector<vector<int>>& matrix) {
        vector<int> result;
        int top = 0, bottom = matrix.size() - 1;
        int left = 0, right = matrix[0].size() - 1;

        while (top <= bottom && left <= right) {
            for (int i = left; i <= right; ++i) result.push_back(matrix[top][i]);
            ++top;

            for (int i = top; i <= bottom; ++i) result.push_back(matrix[i][right]);
            --right;

            if (top <= bottom) {
                for (int i = right; i >= left; --i) result.push_back(matrix[bottom][i]);
                --bottom;
            }

            if (left <= right) {
                for (int i = bottom; i >= top; --i) result.push_back(matrix[i][left]);
                ++left;
            }
        }

        return result;
    }
};
```

## Explanation

### Initialization
- Define four boundaries: `top`, `bottom`, `left`, and `right`.
- Start with the boundaries encompassing the whole matrix.

### Spiral Traversal
1. **Move Left to Right**: Traverse the top row, add elements to the result, and increment `top`.
2. **Move Top to Bottom**: Traverse the right column, add elements to the result, and decrement `right`.
3. **Move Right to Left**: If the `top` boundary is still within the `bottom`, traverse the bottom row, add elements to the result, and decrement `bottom`.
4. **Move Bottom to Top**: If the `left` boundary is still within the `right`, traverse the left column, add elements to the result, and increment `left`.

### Complexity
- **Time Complexity**: \(O(m \times n)\), where \(m\) and \(n\) are the dimensions of the matrix. Each element is visited exactly once.
- **Space Complexity**: \(O(1)\), ignoring the output vector.

## Example Usage
```cpp
#include <iostream>

int main() {
    Solution solution;
    vector<vector<int>> matrix = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    vector<int> result = solution.spiralOrder(matrix);

    for (int num : result) {
        cout << num << " ";
    }
    return 0;
}
```

### Input
Matrix:
```
1  2  3
4  5  6
7  8  9
```

### Output
```
1 2 3 6 9 8 7 4 5
```
