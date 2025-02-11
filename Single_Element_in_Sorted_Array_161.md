
# Solution to "Single Element in a Sorted Array"

This problem can be solved optimally using **binary search** with a time complexity of \( O(\log n) \) and a space complexity of \( O(1) \).

## Problem Link:
[Single Element in a Sorted Array](https://leetcode.com/problems/single-element-in-a-sorted-array/description/)

## Optimized C++ Solution:

```cpp
class Solution {
public:
    int singleNonDuplicate(vector<int>& nums) {
        int left = 0, right = nums.size() - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            // Ensure mid is even for pair checking
            if (mid % 2 == 1) mid--;

            // Check if the pair is broken
            if (nums[mid] == nums[mid + 1]) {
                left = mid + 2; // Move right
            } else {
                right = mid; // Move left
            }
        }
        return nums[left];
    }
};
```

## Explanation:

### Key Observations:
1. The array is sorted, and every element appears exactly twice except for one single element.
2. Using binary search, we can divide the array and locate the single element efficiently.

### Steps:
1. **Binary Search**:
   - Use two pointers: `left` and `right` to perform binary search.
   - Compute the midpoint \( \text{mid} = \text{left} + (\text{right} - \text{left}) / 2 \).
   - Ensure \( \text{mid} \) is even to align with pairs by adjusting \( \text{mid}-- \) if \( \text{mid} \% 2 == 1 \).

2. **Pair Validation**:
   - If \( \text{nums[mid]} == \text{nums[mid+1]} \), the single element must be to the right (inclusive of \( \text{mid+2} \)).
   - Otherwise, the single element must be to the left (inclusive of \( \text{mid} \)).

3. **Termination**:
   - The loop ends when \( \text{left} == \text{right} \), pointing to the single element.

### Complexity Analysis:
- **Time Complexity**: \( O(\log n) \), as the problem size is halved in each step.
- **Space Complexity**: \( O(1) \), no additional space is used.

### Example:
#### Input:
```
nums = [1, 1, 2, 3, 3, 4, 4, 8, 8]
```
#### Execution:
1. Initial range: `left = 0`, `right = 8`
2. Compute `mid = 4`. Check \( \text{nums[mid]} \):
   - \( \text{nums[4]} = 3 \), \( \text{nums[4+1]} = 4 \).
   - Move `right = mid = 4`.
3. Continue until `left == right`.

#### Output:
```
2
```
