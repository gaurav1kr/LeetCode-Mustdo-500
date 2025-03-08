
# Maximal Square - Optimized C++ Solution

## Problem Description
The **Maximal Square** problem from LeetCode can be found [here](https://leetcode.com/problems/maximal-square/description/). The objective is to find the largest square containing only `1`s and return its area.

---

## Optimized Solution

This solution uses a **dynamic programming** approach with space optimization by employing a 1D DP array.

### C++ Code:
```cpp
#include <vector>
#include <algorithm>
using namespace std;

class Solution {
public:
    int maximalSquare(vector<vector<char>>& matrix) {
        int rows = matrix.size();
        if (rows == 0) return 0;
        int cols = matrix[0].size();
        
        // Using a 1D dp array to optimize space
        vector<int> dp(cols + 1, 0);
        int maxSide = 0, prev = 0;
        
        for (int i = 1; i <= rows; ++i) {
            for (int j = 1; j <= cols; ++j) {
                int temp = dp[j];
                if (matrix[i - 1][j - 1] == '1') {
                    dp[j] = min({dp[j - 1], dp[j], prev}) + 1;
                    maxSide = max(maxSide, dp[j]);
                } else {
                    dp[j] = 0;
                }
                prev = temp;
            }
        }
        return maxSide * maxSide;
    }
};
```

---

## Explanation

### Dynamic Programming State:
- `dp[j]` represents the largest side length of the square ending at cell `(i, j)` in the current row.

### Space Optimization:
- Instead of using a 2D `dp` table, we use a 1D array (`dp`) and an additional variable `prev` to store the value of `dp[j - 1]` from the previous row.

### Transition:
- If the current cell `matrix[i-1][j-1]` is `'1'`, then:
  \[
  dp[j] = \min(dp[j-1], dp[j], \text{prev}) + 1
  \]
- Else:
  \[
  dp[j] = 0
  \]

---

## Complexity

- **Time Complexity**: \(O(m \times n)\), where \(m\) is the number of rows and \(n\) is the number of columns.
- **Space Complexity**: \(O(n)\), due to the 1D array `dp`.

---

This solution is both time and space-efficient and works well for large matrices.
