# C++ Solution for "Kth Smallest Element in a Sorted Matrix"

## Problem Description
Given an `n x n` matrix where each of the rows and columns is sorted in ascending order, find the k-th smallest element in the matrix.

### Example:
```cpp
Input: matrix = [
   [1,  5,  9],
   [10, 11, 13],
   [12, 13, 15]
], k = 8
Output: 13
```

### Constraints:
- `n == matrix.length`
- `n == matrix[i].length`
- `1 <= n <= 300`
- `-10^9 <= matrix[i][j] <= 10^9`
- All rows and columns of the matrix are sorted in non-decreasing order.
- `1 <= k <= n^2`

---

## Optimized C++ Solution

### Approach:
1. **Binary Search:** Perform binary search over the possible value range from `matrix[0][0]` to `matrix[n-1][n-1]`.
2. **Count Function:** For a given mid-point, count the number of elements less than or equal to it using the sorted property of the matrix.

### Complexity:
- **Time Complexity:** `O(n * log(max - min))`, where `n` is the size of the matrix and `max` and `min` are the largest and smallest elements in the matrix.
- **Space Complexity:** `O(1)`.

---

### Code:
```cpp
#include <vector>
#include <algorithm>
using namespace std;

class Solution {
public:
    int kthSmallest(vector<vector<int>>& matrix, int k) {
        int n = matrix.size();
        int left = matrix[0][0], right = matrix[n-1][n-1];

        while (left < right) {
            int mid = left + (right - left) / 2;
            int count = countLessEqual(matrix, mid, n);

            if (count < k) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        return left;
    }

private:
    int countLessEqual(const vector<vector<int>>& matrix, int target, int n) {
        int count = 0, col = n - 1;

        for (int row = 0; row < n; ++row) {
            while (col >= 0 && matrix[row][col] > target) {
                --col;
            }
            count += (col + 1);
        }

        return count;
    }
};
```

---

### Explanation:
1. **Binary Search:**
   - Use `left` and `right` to define the range of possible values.
   - Compute the midpoint and count elements <= midpoint.
   - Narrow the search range based on the count.

2. **Efficient Counting:**
   - Start from the top-right corner of the matrix.
   - Move left while the current element is greater than the target.
   - Count all valid elements row-wise.

### Key Points:
- The solution leverages the sorted properties of rows and columns for efficient counting.
- Binary search reduces the time complexity compared to a brute-force approach.
