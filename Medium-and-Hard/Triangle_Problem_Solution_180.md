# Triangle Problem Solution

This document provides a concise and optimized solution for the Triangle problem from [LeetCode](https://leetcode.com/problems/triangle/description/).

## Problem Description
Given a triangle array, return the minimum path sum from top to bottom. Each step, you may move to adjacent numbers on the row below.

### Example:
Input:
```
[[2],
 [3,4],
 [6,5,7],
 [4,1,8,3]]
```
Output:
```
11 (2 + 3 + 5 + 1)
```

## Optimized C++ Solution
Here is a concise and efficient solution using the bottom-up dynamic programming approach:

```cpp
#include <vector>
using namespace std;

class Solution {
public:
    int minimumTotal(vector<vector<int>>& triangle) {
        for (int row = triangle.size() - 2; row >= 0; --row) {
            for (int col = 0; col < triangle[row].size(); ++col) {
                triangle[row][col] += min(triangle[row + 1][col], triangle[row + 1][col + 1]);
            }
        }
        return triangle[0][0];
    }
};
```

### Explanation
1. **Bottom-Up Approach**: Start from the second-last row and move upwards, updating each element with the minimum path sum from that point.
2. **In-Place Modification**: The triangle array is updated in place to reduce space complexity.
3. **Return Value**: The minimum path sum is stored at the top element of the triangle (`triangle[0][0]`).

### Complexity Analysis
- **Time Complexity**: \(O(n^2)\), where \(n\) is the number of rows in the triangle.
- **Space Complexity**: \(O(1)\), as the input triangle is reused for calculations.

This solution is efficient and avoids additional memory allocation for a separate DP table.

## Notes
- Ensure that the input triangle is mutable since this solution modifies it in place.
- For immutable inputs, consider using a separate DP array to store the calculations.

For any questions or improvements, feel free to reach out!
