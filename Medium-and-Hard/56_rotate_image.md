## Rotate Image - Optimal C++ Solution

### Problem Statement
Given an `n x n` 2D matrix representing an image, rotate the image by **90 degrees (clockwise)** in place.

### Approach
The most efficient way to achieve this in-place rotation is:
1. **Transpose the matrix**: Convert rows into columns by swapping `matrix[i][j]` with `matrix[j][i]` for `i < j`.
2. **Reverse each row**: This mirrors the matrix horizontally, completing the 90-degree rotation.

### C++ Solution
```cpp
#include <vector>
using namespace std;

class Solution {
public:
    void rotate(vector<vector<int>>& matrix) {
        int n = matrix.size();
        
        // Step 1: Transpose the matrix
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                swap(matrix[i][j], matrix[j][i]);
            }
        }
        
        // Step 2: Reverse each row
        for (int i = 0; i < n; ++i) {
            reverse(matrix[i].begin(), matrix[i].end());
        }
    }
};
```

### Complexity Analysis
- **Transpose Step:** \(O(n^2)\) - Since we iterate over roughly half the elements.
- **Reverse Rows Step:** \(O(n^2)\) - Each row is reversed in \(O(n)\) time.
- **Total Time Complexity:** \(O(n^2)\)
- **Space Complexity:** \(O(1)\) - In-place modification without extra space.

### Example
#### **Input:**
```cpp
[[1,2,3],
 [4,5,6],
 [7,8,9]]
```
#### **Output:**
```cpp
[[7,4,1],
 [8,5,2],
 [9,6,3]]
```

### Summary
- This is an efficient in-place solution with **O(1) extra space**.
- Uses **transpose + reverse** technique to achieve rotation.
- Suitable for competitive programming and interviews.
