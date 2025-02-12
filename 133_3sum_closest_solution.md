# 3Sum Closest - C++ Solution

## Problem Description
You can find the problem description here: [LeetCode - 3Sum Closest](https://leetcode.com/problems/3sum-closest/description/)

## Optimized C++ Solution

```cpp
#include <vector>
#include <algorithm>
#include <cmath>
using namespace std;

class Solution {
public:
    int threeSumClosest(vector<int>& nums, int target) {
        sort(nums.begin(), nums.end());
        int closest = nums[0] + nums[1] + nums[2];
        
        for (int i = 0; i < nums.size() - 2; ++i) {
            int left = i + 1, right = nums.size() - 1;
            while (left < right) {
                int sum = nums[i] + nums[left] + nums[right];
                if (abs(target - sum) < abs(target - closest))
                    closest = sum;
                
                if (sum < target)
                    ++left;
                else if (sum > target)
                    --right;
                else
                    return sum; // Exact match found
            }
        }
        return closest;
    }
};
```

## Explanation

### 1. Sorting the Array
The array is sorted to facilitate the two-pointer approach.

### 2. Two-Pointer Approach
- Fix one number (`nums[i]`) and use two pointers (`left` and `right`) to find the other two numbers.
- Adjust pointers (`left` or `right`) based on whether the current sum is less than or greater than the target.

### 3. Closest Sum Tracking
Maintain a variable `closest` to store the closest sum found so far.

### 4. Early Termination
If an exact match (`sum == target`) is found, the function immediately returns the sum.

## Complexity Analysis

### Time Complexity:
- Sorting: \(O(n \log n)\)
- Two-pointer loop: \(O(n^2)\)
- Overall: \(O(n^2)\)

### Space Complexity:
- \(O(1)\), as no extra space is used apart from variables.

## Notes
This solution is optimal and widely used for solving this problem efficiently.
