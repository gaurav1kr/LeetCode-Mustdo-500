## House Robber II Solution

This is a concise and optimized C++ solution for the **House Robber II** problem on LeetCode.

### Problem Description

The houses are arranged in a circle. Adjacent houses cannot be robbed, meaning the first and last house cannot be robbed together. The task is to maximize the robbery amount.

---

### C++ Code

```cpp
#include <vector>
#include <algorithm>
using namespace std;

class Solution {
public:
    int rob(vector<int>& nums) {
        int n = nums.size();
        if (n == 1) return nums[0];
        return max(robRange(nums, 0, n - 2), robRange(nums, 1, n - 1));
    }

private:
    int robRange(const vector<int>& nums, int start, int end) {
        int prev1 = 0, prev2 = 0;
        for (int i = start; i <= end; ++i) {
            int temp = max(prev1, prev2 + nums[i]);
            prev2 = prev1;
            prev1 = temp;
        }
        return prev1;
    }
};
```

---

### Explanation

#### Key Idea

- In this problem, houses are arranged in a circle, so we can't rob the first and last house together. To handle this, we split the problem into two subproblems:
  - Rob houses `[0...n-2]`.
  - Rob houses `[1...n-1]`.
- The result is the maximum of these two scenarios.

#### `robRange` Function

- A helper function that calculates the maximum amount for a linear arrangement of houses using dynamic programming.
- Uses two variables (`prev1` and `prev2`) to keep track of the maximum amounts for optimized space complexity.

#### Edge Case

- If there is only one house (`n == 1`), return the value of that house directly.

---

### Complexity

- **Time Complexity**: O(n), where `n` is the number of houses.
- **Space Complexity**: O(1), as we use only two variables to store intermediate results.

---

### Example Input/Output

#### Example 1:

**Input**:
```cpp
nums = [2, 3, 2]
```

**Output**:
```cpp
3
```

**Explanation**:
- Rob houses `[0...n-2]`: Max amount is `2`.
- Rob houses `[1...n-1]`: Max amount is `3`.
- Result: `max(2, 3) = 3`.

#### Example 2:

**Input**:
```cpp
nums = [1, 2, 3, 1]
```

**Output**:
```cpp
4
```

**Explanation**:
- Rob houses `[0...n-2]`: Max amount is `4`.
- Rob houses `[1...n-1]`: Max amount is `4`.
- Result: `max(4, 4) = 4`.

---

### Notes

This solution is both efficient and clean, leveraging dynamic programming with reduced space complexity. It ensures that we maximize the robbery amount while adhering to the problem's constraints.
