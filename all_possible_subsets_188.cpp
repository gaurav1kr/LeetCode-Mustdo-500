# Subsets II - Optimized C++ Solution

## Problem Description
The problem **"Subsets II"** requires generating all possible subsets of a given integer array `nums`, where the array may contain duplicates. The solution must avoid duplicate subsets.

You can find the problem statement here: [LeetCode - Subsets II](https://leetcode.com/problems/subsets-ii/description/)

## Optimized C++ Solution
Below is an optimized C++ implementation of the solution using **backtracking**:

```cpp
#include <vector>
#include <algorithm>
using namespace std;

class Solution 
{
public:
    void backtrack(vector<int>& nums, int start, vector<int>& current, vector<vector<int>>& result) 
    {
        result.push_back(current); // Add the current subset to the result
        
        for (int i = start; i < nums.size(); ++i) 
	{
            // Skip duplicates
            if (i > start && nums[i] == nums[i - 1]) continue;

            // Include nums[i] in the current subset
            current.push_back(nums[i]);
            
            // Recurse for the next element
            backtrack(nums, i + 1, current, result);
            
            // Backtrack by removing the last element
            current.pop_back();
        }
    }
    
    vector<vector<int>> subsetsWithDup(vector<int>& nums) 
    {
        vector<vector<int>> result;
        vector<int> current;

        // Sort the array to handle duplicates easily
        sort(nums.begin(), nums.end());
        
        // Start backtracking
        backtrack(nums, 0, current, result);
        
        return result;
    }
};
```

## Explanation

### Key Steps in the Solution

1. **Sorting the Input Array**:
   - Sorting ensures that duplicate elements are adjacent. This simplifies the process of skipping duplicates.

2. **Backtracking**:
   - The `backtrack` function explores all possible subsets.
   - At each recursion step:
     - Add the current subset (`current`) to the `result`.
     - Iterate through the remaining elements, adding each to the `current` subset and recursing further.

3. **Skipping Duplicates**:
   - To avoid duplicate subsets, skip the current element (`nums[i]`) if it is the same as the previous element (`nums[i - 1]`) and not the first element in the current loop (`i > start`).

4. **Backtracking Step**:
   - After exploring the subsets that include `nums[i]`, remove `nums[i]` from `current` (backtrack) to explore other possibilities.

### Complexity Analysis

- **Time Complexity**: 
  - \(O(2^n)\), as there are \(2^n\) possible subsets.

- **Space Complexity**: 
  - \(O(n)\), due to the recursion stack and `current` subset storage.

---

This approach is efficient and avoids generating duplicate subsets by leveraging sorting and skipping duplicate elements during recursion.
