# Jump Game - C++ Solution

## Problem Statement
Given an array of non-negative integers `nums`, where `nums[i]` represents the maximum jump length from index `i`, determine if you can reach the last index.

## Optimal C++ Solution

```cpp
class Solution {
public:
    bool canJump(vector<int>& nums) {
        int maxReach = 0;
        for (int i = 0; i < nums.size(); i++) {
            if (i > maxReach) return false;
            maxReach = max(maxReach, i + nums[i]);
        }
        return true;
    }
};
```

## Explanation
- We track the maximum index we can reach (`maxReach`).
- If `i` ever exceeds `maxReach`, we return `false` as we are stuck.
- Otherwise, we update `maxReach = max(maxReach, i + nums[i])`.
- The solution runs in **O(n) time** and **O(1) space**, making it optimal.

## Complexity Analysis
- **Time Complexity:** `O(n)`, as we iterate through the array once.
- **Space Complexity:** `O(1)`, as we use only a single integer variable.

---
LeetCode Problem: [Jump Game](https://leetcode.com/problems/jump-game)
