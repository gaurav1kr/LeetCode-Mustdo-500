# Leetcode - Permutations (C++ Solution)

## Solution (Backtracking Approach)
```cpp
class Solution {
public:
    vector<vector<int>> permute(vector<int>& nums) {
        vector<vector<int>> res;
        backtrack(nums, 0, res);
        return res;
    }

    void backtrack(vector<int>& nums, int start, vector<vector<int>>& res) {
        if (start == nums.size()) {
            res.push_back(nums);
            return;
        }
        for (int i = start; i < nums.size(); ++i) {
            swap(nums[start], nums[i]);
            backtrack(nums, start + 1, res);
            swap(nums[start], nums[i]);  // backtrack
        }
    }
};
```

## Explanation
- Uses **backtracking** to generate all possible permutations.
- Swaps elements to explore different configurations.
- Reverts swaps (backtracking) to explore new possibilities.
- **Time Complexity:** \(O(N!)\) (since we generate all permutations)
- **Space Complexity:** \(O(N!)\) (for storing all permutations)
