# Find Peak Element - Optimized C++ Solution

This is an optimized solution for the **Find Peak Element** problem on LeetCode, implemented using Binary Search.

## Problem Link
[Find Peak Element](https://leetcode.com/problems/find-peak-element/description/)

## C++ Solution

```cpp
class Solution {
public:
    int findPeakElement(vector<int>& nums) {
        int left = 0, right = nums.size() - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] > nums[mid + 1])
                right = mid; // Peak is in the left half (including mid)
            else
                left = mid + 1; // Peak is in the right half
        }
        return left; // Left and right converge to the peak index
    }
};
```

## Explanation

### Approach
- **Binary Search**:
  - The goal is to find a peak element where `nums[i] > nums[i+1]` and `nums[i] > nums[i-1]`.
  - Compare `nums[mid]` with `nums[mid + 1]`:
    - If `nums[mid] > nums[mid + 1]`, it means a peak exists in the left half (including `mid`).
    - Otherwise, the peak must exist in the right half.
  - Adjust the `left` or `right` pointers accordingly.

### Complexity
- **Time Complexity**: `O(log n)`
  - Each iteration halves the search space.
- **Space Complexity**: `O(1)`
  - Operates directly on the input array without extra space.

---

This solution is efficient and adheres to the constraints of the problem.