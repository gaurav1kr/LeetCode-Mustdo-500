
# Shortest Unsorted Continuous Subarray

This document provides an optimized and concise C++ solution to the LeetCode problem: [Shortest Unsorted Continuous Subarray](https://leetcode.com/problems/shortest-unsorted-continuous-subarray/description/).

## Problem Approach
The goal is to determine the boundaries of the subarray that needs sorting. This is done by identifying where the sorting order is violated from both the left and right sides of the array.

## C++ Solution
```cpp
#include <vector>
#include <algorithm>
#include <climits>

class Solution {
public:
    int findUnsortedSubarray(std::vector<int>& nums) {
        int n = nums.size();
        int start = -1, end = -1;
        int maxVal = INT_MIN, minVal = INT_MAX;

        // Traverse from left to right to find the right boundary
        for (int i = 0; i < n; ++i) {
            if (nums[i] < maxVal) end = i;
            else maxVal = nums[i];
        }

        // Traverse from right to left to find the left boundary
        for (int i = n - 1; i >= 0; --i) {
            if (nums[i] > minVal) start = i;
            else minVal = nums[i];
        }

        return (start == -1) ? 0 : (end - start + 1);
    }
};
```

## Explanation
1. **Left-to-right pass**:
   - Track the maximum value encountered so far (`maxVal`).
   - If the current element is smaller than `maxVal`, update the `end` index.
2. **Right-to-left pass**:
   - Track the minimum value encountered so far (`minVal`).
   - If the current element is larger than `minVal`, update the `start` index.
3. If no violations are found (`start == -1`), the array is already sorted, and we return `0`.

## Complexity
- **Time Complexity**: \(O(n)\), where \(n\) is the size of the array, as we perform two linear scans.
- **Space Complexity**: \(O(1)\), as no extra space is used.

This solution is efficient and solves the problem with minimal overhead.
