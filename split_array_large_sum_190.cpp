# Solution to "Split Array Largest Sum"

This document provides the optimized C++ solution to the LeetCode problem [Split Array Largest Sum](https://leetcode.com/problems/split-array-largest-sum/description/).

## Problem Description

Given an array `nums` and an integer `m`, split the array into `m` non-empty continuous subarrays such that the largest sum among these subarrays is minimized. Return that minimized largest sum.

---

## Optimized C++ Solution

```cpp
#include <vector>
#include <numeric>
#include <algorithm>
#include <iostream>

class Solution {
public:
    bool canSplit(const std::vector<int>& nums, int maxSum, int m) 
    {
        int currentSum = 0, splits = 1;
        for (int num : nums) 
	{
            if (currentSum + num > maxSum) 
	    {
                currentSum = num; // Start a new subarray
                splits++;
                if (splits > m) return false; // Exceeds allowed splits
            } 
	    else 
	    {
                currentSum += num;
            }
        }
        return true;
    }
    
    int splitArray(std::vector<int>& nums, int m) 
    {
        int left = *std::max_element(nums.begin(), nums.end()); // Minimum possible largest sum
        int right = std::accumulate(nums.begin(), nums.end(), 0); // Maximum possible largest sum
        
        while (left < right) 
	{
            int mid = left + (right - left) / 2;
            if (canSplit(nums, mid, m)) 
	    {
                right = mid; // Try for a smaller largest sum
            } 
	    else 
	    {
                left = mid + 1; // Increase the largest sum
            }
        }
        
        return left;
    }
};

// Example usage
int main() {
    Solution solution;
    std::vector<int> nums = {7, 2, 5, 10, 8};
    int m = 2;
    std::cout << "The minimized largest sum is: " << solution.splitArray(nums, m) << std::endl;
    return 0;
}
```

---

## Explanation

### 1. **Binary Search**

- The search range is between the largest single element (`left`) and the sum of all elements (`right`).
- The middle point (`mid`) represents a candidate for the largest sum.

### 2. **Greedy Check (`canSplit` Function)**

- Iterate through the array to check if it can be split into `m` or fewer subarrays where the sum of each subarray does not exceed `mid`.
- If the sum exceeds `mid`, start a new subarray and count the number of splits.

### 3. **Adjusting the Binary Search Range**

- If `canSplit` is true, try for a smaller largest sum by moving `right` to `mid`.
- Otherwise, move `left` to `mid + 1`.

---

## Complexity

- **Time Complexity**: 
  - \(O(n \log(\text{sum}))\), where \(n\) is the size of the array, and \\(\text{sum}\\) is the range of binary search (\\(\text{right} - \text{left}\\)).
- **Space Complexity**: 
  - \(O(1)\), as no extra space is used apart from variables.

---

## Example Usage

### Input:
```plaintext
nums = [7, 2, 5, 10, 8], m = 2
```

### Output:
```plaintext
The minimized largest sum is: 18
```

---
