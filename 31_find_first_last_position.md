# Approach for Finding First and Last Position of Element in Sorted Array

## Problem Statement
Given a sorted array of integers `nums` and a target value `target`, find the starting and ending position of `target` in `nums`. If `target` is not found, return `{-1, -1}`.

## Approach
To solve this problem efficiently, we use **binary search**. The brute force approach of scanning the entire array would take O(n) time, but binary search allows us to find the positions in **O(log n)** time.

### Steps:
1. **Find the First Occurrence:**  
   - Perform binary search on `nums`.
   - If `nums[mid]` is greater than or equal to `target`, move `right` to `mid - 1`.
   - Otherwise, move `left` to `mid + 1`.
   - If `nums[mid] == target`, store `mid` as a potential result.

2. **Find the Last Occurrence:**  
   - Perform binary search again.
   - If `nums[mid]` is less than or equal to `target`, move `left` to `mid + 1`.
   - Otherwise, move `right` to `mid - 1`.
   - If `nums[mid] == target`, store `mid` as a potential result.

### Time Complexity
- Each binary search takes **O(log n)**.
- Since we perform two binary searches, the total complexity is **O(2 log n) ≈ O(log n)**.

## C++ Code
```cpp
class Solution {
public:
    vector<int> searchRange(vector<int>& nums, int target) {
        return {findFirst(nums, target), findLast(nums, target)};
    }

private:
    int findFirst(vector<int>& nums, int target) {
        int left = 0, right = nums.size() - 1, result = -1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] >= target) right = mid - 1;
            else left = mid + 1;
            if (nums[mid] == target) result = mid;
        }
        return result;
    }

    int findLast(vector<int>& nums, int target) {
        int left = 0, right = nums.size() - 1, result = -1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] <= target) left = mid + 1;
            else right = mid - 1;
            if (nums[mid] == target) result = mid;
        }
        return result;
    }
};
```
