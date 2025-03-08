
# Burst Balloons Problem - Optimized C++ Solution

## Problem Summary:
Given an array of balloons, you can burst a balloon `i` to earn coins equal to `nums[left] * nums[i] * nums[right]`, where `left` and `right` are the indices of the balloons adjacent to `i`. The goal is to maximize the total coins obtained by bursting the balloons.

---

## Optimized C++ Solution
```cpp
#include <vector>
#include <algorithm>
using namespace std;

class Solution {
public:
    int maxCoins(vector<int>& nums) {
        int n = nums.size();
        
        // Add virtual balloons with value 1 at both ends
        nums.insert(nums.begin(), 1);
        nums.push_back(1);
        
        // DP table
        vector<vector<int>> dp(n + 2, vector<int>(n + 2, 0));
        
        // Fill the DP table
        for (int len = 1; len <= n; ++len) { // Length of the subarray
            for (int left = 1; left <= n - len + 1; ++left) {
                int right = left + len - 1;
                
                // Find the maximum coins for the subarray nums[left:right]
                for (int k = left; k <= right; ++k) {
                    // Coins gained by bursting balloon k last
                    int coins = nums[left - 1] * nums[k] * nums[right + 1];
                    coins += dp[left][k - 1]; // Left subarray
                    coins += dp[k + 1][right]; // Right subarray
                    dp[left][right] = max(dp[left][right], coins);
                }
            }
        }
        
        // The result is the maximum coins for the entire array
        return dp[1][n];
    }
};
```

---

## Explanation

### 1. **State Definition**:
   - `dp[left][right]` represents the maximum coins that can be obtained by bursting all the balloons in the range `[left, right]`.

### 2. **State Transition**:
   - If `k` is the last balloon to burst in the range `[left, right]`, the coins gained are:
     \[
     dp[left][right] = \text{nums}[left - 1] \times \text{nums}[k] \times \text{nums}[right + 1] + dp[left][k-1] + dp[k+1][right]
     \]
   - Update `dp[left][right]` by iterating over all possible positions of `k` in `[left, right]`.

### 3. **Base Case**:
   - For ranges of length `0` (empty ranges), `dp[left][right] = 0`.

### 4. **Boundary Conditions**:
   - Add virtual balloons with value `1` at the beginning and end of the array to handle edge cases.

### 5. **Time Complexity**:
   - The DP table has \(O(n^2)\) states, and for each state, we iterate over \(O(n)\) possible values of `k`. Thus, the time complexity is \(O(n^3)\).

### 6. **Space Complexity**:
   - The DP table requires \(O(n^2)\) space.

---

## Example Usage
```cpp
int main() {
    Solution solution;
    vector<int> balloons = {3, 1, 5, 8};
    int result = solution.maxCoins(balloons);
    // Output: 167
    cout << "Maximum coins: " << result << endl;
    return 0;
}
```

### Example Input:
```plaintext
nums = [3, 1, 5, 8]
```

### Example Output:
```plaintext
Maximum coins: 167
```
