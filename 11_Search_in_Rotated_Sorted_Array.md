# Search in Rotated Sorted Array

## Problem Description
Given the array `nums` sorted in ascending order (with distinct values) and then potentially rotated at an unknown pivot index, search for a given target value. If the target exists, return its index. Otherwise, return `-1`.

You must write an algorithm with \(O(\log n)\) runtime complexity.

### Example Input:
```cpp
nums = [4, 5, 6, 7, 0, 1, 2]
target = 0
```

### Example Output:
```cpp
Output: 4
```

## Optimal C++ Solution

```cpp
#include <vector>
using namespace std;

class Solution {
public:
    int search(vector<int>& nums, int target) {
        int left = 0, right = nums.size() - 1;

        while (left <= right) {
            int mid = left + (right - left) / 2;

            if (nums[mid] == target) {
                return mid;
            }

            // Check which part is sorted
            if (nums[left] <= nums[mid]) { // Left part is sorted
                if (nums[left] <= target && target < nums[mid]) {
                    right = mid - 1; // Target is in the left part
                } else {
                    left = mid + 1; // Target is in the right part
                }
            } else { // Right part is sorted
                if (nums[mid] < target && target <= nums[right]) {
                    left = mid + 1; // Target is in the right part
                } else {
                    right = mid - 1; // Target is in the left part
                }
            }
        }

        return -1; // Target not found
    }
};
```

## Explanation
1. **Binary Search Modification**:
   - At every iteration, check if the middle element matches the target.
   - Determine which half (left or right) of the array is sorted.
   - Use the sorted property to decide which half to search in.

2. **Key Observations**:
   - If `nums[left] <= nums[mid]`, then the left part is sorted.
   - If `nums[mid] <= nums[right]`, then the right part is sorted.
   - Based on where the target lies, adjust `left` or `right`.

### Example Run
Input:
```cpp
vector<int> nums = {4, 5, 6, 7, 0, 1, 2};
int target = 0;
Solution sol;
int result = sol.search(nums, target); // Output: 4
```

## Complexity Analysis
- **Time Complexity**: \(O(\log n)\), as the solution uses binary search.
- **Space Complexity**: \(O(1)\), since no extra space is used.

This implementation ensures an efficient search by leveraging the properties of the rotated sorted array.
