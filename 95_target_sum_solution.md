# Optimized C++ Solution for LeetCode Problem: Target Sum

The problem **"Target Sum"** on LeetCode can be solved efficiently using **Dynamic Programming**. Below is an optimized C++ solution that utilizes a bottom-up DP approach:

## Optimized C++ Solution:

```cpp
#include <vector>
#include <numeric>
#include <iostream>

using namespace std;

class Solution {
public:
    int findTargetSumWays(vector<int>& nums, int target) {
        int totalSum = accumulate(nums.begin(), nums.end(), 0);

        // If the target is out of the possible sum range, return 0
        if (totalSum < abs(target) || (totalSum + target) % 2 != 0) {
            return 0;
        }

        // Transform the problem into a subset sum problem
        int sum = (totalSum + target) / 2;

        return countSubsets(nums, sum);
    }

private:
    int countSubsets(const vector<int>& nums, int sum) {
        vector<int> dp(sum + 1, 0);
        dp[0] = 1; // Base case: There's one way to make sum 0, which is using no elements.

        for (int num : nums) {
            for (int j = sum; j >= num; --j) {
                dp[j] += dp[j - num];
            }
        }

        return dp[sum];
    }
};
```

## Explanation:

### Problem Transformation:
- The problem can be converted to a subset sum problem:
  - Let `P` be the subset of elements assigned `+` and `N` be the subset assigned `-`.
  - The goal is to find subsets such that \( P - N = \text{target} \).
  - This can be rewritten as \( P + N = \text{totalSum} \) and \( P = (\text{totalSum} + \text{target}) / 2 \).

### Subset Sum:
- The problem reduces to finding the number of subsets whose sum equals \( (\text{totalSum} + \text{target}) / 2 \).

### Dynamic Programming:
- Use a 1D DP array `dp`, where `dp[j]` represents the number of ways to form a subset with a sum of `j`.
- Update the DP array in reverse order to avoid overwriting results from the current iteration.

### Edge Cases:
- If \( \text{totalSum} < \text{abs(target)} \), it's impossible to reach the target, so return 0.
- If \( (\text{totalSum} + \text{target}) \) is odd, the result is also 0.

## Complexity:
- **Time Complexity**: \( O(n \cdot \text{sum}) \), where \( n \) is the size of the `nums` array and `sum` is the transformed subset sum.
- **Space Complexity**: \( O(\text{sum}) \), as we use a 1D DP array.

## Example Usage:

```cpp
int main() {
    Solution sol;
    vector<int> nums = {1, 1, 1, 1, 1};
    int target = 3;
    cout << sol.findTargetSumWays(nums, target) << endl; // Output: 5
    return 0;
}
```

This solution is efficient and works well for the constraints of the problem.
