
# Optimal Solution for Jump Game II

## Problem Description
[LeetCode Problem: Jump Game II](https://leetcode.com/problems/jump-game-ii/description/)

The goal is to find the minimum number of jumps needed to reach the last index of the array.

---

## C++ Solution
This solution uses a **greedy algorithm** to achieve optimal performance with an \(O(n)\) time complexity.

```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    int jump(std::vector<int>& nums) {
        int n = nums.size();
        if (n <= 1) return 0;

        int jumps = 0;       // Number of jumps taken
        int currentEnd = 0;  // Farthest we can reach with the current jump
        int farthest = 0;    // Farthest we can reach with the next jump

        for (int i = 0; i < n - 1; ++i) {
            farthest = std::max(farthest, i + nums[i]);

            // If we've reached the end of the range for the current jump
            if (i == currentEnd) {
                jumps++;
                currentEnd = farthest;

                // If we can already reach the last index, stop
                if (currentEnd >= n - 1) break;
            }
        }

        return jumps;
    }
};
```

---

## Explanation
1. **Greedy Choice:**
   - At each step, calculate the `farthest` index we can reach.

2. **Jump Decision:**
   - When the current index reaches the `currentEnd`, it means we need a jump to proceed further.
   - Update `currentEnd` to `farthest` to continue.

3. **Early Exit:**
   - If `currentEnd` reaches or surpasses the last index, we stop iterating since we've already found the minimum jumps.

---

## Complexity Analysis
- **Time Complexity:** \(O(n)\), where \(n\) is the size of the input array. We traverse the array once.
- **Space Complexity:** \(O(1)\), as no extra space is used apart from variables.

---

## Example Usage
```cpp
int main() {
    Solution sol;
    std::vector<int> nums = {2, 3, 1, 1, 4};
    int result = sol.jump(nums); // Output: 2
    return 0;
}
```

### Input:
`nums = [2, 3, 1, 1, 4]`

### Output:
`2`

---

This solution is efficient and well-suited for large inputs.
