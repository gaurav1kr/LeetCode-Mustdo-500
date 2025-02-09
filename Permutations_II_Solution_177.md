# LeetCode Problem: Permutations II

This document contains the optimized and concise C++ solution for solving the LeetCode problem **Permutations II**. This problem requires generating all unique permutations of an array of integers, which may contain duplicates.

## Problem Link
[Permutations II - LeetCode](https://leetcode.com/problems/permutations-ii/description/)

## C++ Solution
```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    std::vector<std::vector<int>> permuteUnique(std::vector<int>& nums) {
        std::vector<std::vector<int>> results;
        std::vector<int> current;
        std::vector<bool> used(nums.size(), false);
        std::sort(nums.begin(), nums.end()); // Sort to handle duplicates
        backtrack(nums, used, current, results);
        return results;
    }

private:
    void backtrack(const std::vector<int>& nums, std::vector<bool>& used,
                   std::vector<int>& current, std::vector<std::vector<int>>& results) {
        if (current.size() == nums.size()) {
            results.push_back(current);
            return;
        }
        for (int i = 0; i < nums.size(); ++i) {
            if (used[i] || (i > 0 && nums[i] == nums[i - 1] && !used[i - 1])) continue;
            used[i] = true;
            current.push_back(nums[i]);
            backtrack(nums, used, current, results);
            used[i] = false;
            current.pop_back();
        }
    }
};
```

## Key Features of the Solution

### 1. **Sorting to Handle Duplicates**
- The input array is sorted so that duplicate numbers are adjacent.
- The condition `(i > 0 && nums[i] == nums[i - 1] && !used[i - 1])` ensures that duplicates are only processed once in a given recursive branch.

### 2. **Backtracking**
- The algorithm explores all possible permutations by iteratively including unused elements.
- Used elements are tracked using the `used` boolean vector.

### 3. **Efficiency**
- Sorting takes \(O(n \log n)\), and generating permutations takes \(O(n!)\) in the worst case, making this approach efficient for the problem size.

### 4. **Compact Code**
- The logic is implemented in a few lines, balancing readability and performance.

## Example Usage
```cpp
#include <iostream>

int main() {
    Solution sol;
    std::vector<int> nums = {1, 1, 2};
    auto result = sol.permuteUnique(nums);
    for (const auto& perm : result) {
        for (int num : perm) {
            std::cout << num << " ";
        }
        std::cout << "\n";
    }
    return 0;
}
```

### Output for `[1, 1, 2]`:
```
1 1 2
1 2 1
2 1 1
```

This ensures that all permutations are unique and efficiently generated.
