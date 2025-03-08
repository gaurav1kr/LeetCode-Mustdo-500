
# Partition Equal Subset Sum Solution

## Problem Statement
The problem is to determine if the given array can be partitioned into two subsets such that the sums of the subsets are equal. 

You can find the full problem description on LeetCode: [Partition Equal Subset Sum](https://leetcode.com/problems/partition-equal-subset-sum/description/)

---

## Optimal C++ Solution

### Explanation
1. The target sum for each subset will be `totalSum / 2`. If `totalSum` is odd, partitioning is impossible.
2. We use a **1D dynamic programming (DP) array**, where `dp[j]` indicates whether a subset with sum `j` can be formed using elements of the array.
3. The array is updated in reverse order during each iteration to avoid overwriting values from the current step.

---

### Code
```cpp
#include <vector>
#include <numeric>
using namespace std;

class Solution {
public:
    bool canPartition(vector<int>& nums) {
        int totalSum = accumulate(nums.begin(), nums.end(), 0);

        // If the total sum is odd, we cannot partition into two equal subsets
        if (totalSum % 2 != 0) {
            return false;
        }

        int target = totalSum / 2;
        vector<bool> dp(target + 1, false);
        dp[0] = true; // Base case: subset with sum 0 is always possible

        for (int num : nums) {
            for (int j = target; j >= num; --j) {
                dp[j] = dp[j] || dp[j - num];
            }
        }

        return dp[target];
    }
};
```

---

### Key Points
- **Time Complexity**: \(O(n \times \text{target})\), where \(n\) is the size of the array, and \(\text{target} = \text{totalSum} / 2\).
- **Space Complexity**: \(O(\text{target})\), since we use a 1D DP array.
- **Optimization**: Iterating the `dp` array in reverse prevents overwriting values from the current iteration.

---

### Example Usage
```cpp
#include <iostream>
int main() {
    Solution solution;
    vector<int> nums = {1, 5, 11, 5};
    cout << (solution.canPartition(nums) ? "True" : "False") << endl; // Output: True
    return 0;
}
```

---

### Notes
This solution is efficient and provides an optimal approach for solving the Partition Equal Subset Sum problem using dynamic programming.
