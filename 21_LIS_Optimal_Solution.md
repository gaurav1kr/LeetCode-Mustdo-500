# Longest Increasing Subsequence (LIS) - Optimal C++ Solution

## Problem Statement
Given an integer array `nums`, return the length of the longest strictly increasing subsequence.

## Optimal C++ Solution (O(n log n))

```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    int lengthOfLIS(std::vector<int>& nums) {
        std::vector<int> lis;
        for (int num : nums) {
            auto it = std::lower_bound(lis.begin(), lis.end(), num);
            if (it == lis.end()) lis.push_back(num);
            else *it = num;
        }
        return lis.size();
    }
};
```

## Explanation:
1. **Uses a `vector<int> lis`**: Maintains an array to represent the smallest possible increasing subsequence.
2. **Binary search (`lower_bound`)**:  
   - If `num` is greater than the largest element in `lis`, append it.  
   - Otherwise, replace the first element in `lis` that is **≥ `num`** to keep it minimal.  
3. **Time Complexity: `O(n log n)`** due to `lower_bound()` usage.  
4. **Space Complexity: `O(n)`** (in the worst case, all elements are part of LIS).  

This is the shortest and most optimal solution for LIS. 🚀
