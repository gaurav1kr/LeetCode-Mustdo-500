
# Find Minimum in Rotated Sorted Array - Optimized C++ Solution

## Problem Statement
The problem is to find the minimum element in a rotated sorted array. The array was originally sorted in ascending order, but then it was rotated at some unknown pivot.

You are guaranteed that:
- All elements are unique.
- The array does not contain duplicates.
- The array is non-empty.

### Example
Input: `nums = [4, 5, 6, 7, 0, 1, 2]`  
Output: `0`

---

## Optimized C++ Solution
To solve this problem efficiently, we use a binary search approach with a time complexity of \(O(\log n)\).

### Code
```cpp
#include <vector>
#include <iostream>
using namespace std;

class Solution {
public:
    int findMin(vector<int>& nums) {
        int left = 0, right = nums.size() - 1;

        // If the array is not rotated
        if (nums[left] <= nums[right]) {
            return nums[left];
        }

        while (left < right) {
            int mid = left + (right - left) / 2;

            // If mid element is greater than the rightmost element,
            // the minimum must be in the right part of the array.
            if (nums[mid] > nums[right]) {
                left = mid + 1;
            } 
            // Otherwise, the minimum is in the left part (including mid).
            else {
                right = mid;
            }
        }

        // At the end, left == right, which points to the minimum element.
        return nums[left];
    }
};

// Example usage
int main() {
    Solution solution;
    vector<int> nums = {4, 5, 6, 7, 0, 1, 2};
    cout << "The minimum is: " << solution.findMin(nums) << endl;
    return 0;
}
```

---

### Explanation
1. **Binary Search Logic**:
   - Check the middle element \( \text{nums[mid]} \).
   - If \( \text{nums[mid]} > \text{nums[right]} \), the minimum is in the right half of the array because the rotation point lies there.
   - Otherwise, the minimum is in the left half (including the middle element).

2. **Edge Cases**:
   - If the array is already sorted (\( \text{nums[left]} \leq \text{nums[right]} \)), return the first element directly.
   - Handles arrays with only one element, as the loop condition ensures correctness.

3. **Time Complexity**:
   - \(O(\log n)\), because the search space is halved in each step.

4. **Space Complexity**:
   - \(O(1)\), as no extra space is used.

---

### Output for Example Input
Input: `nums = [4, 5, 6, 7, 0, 1, 2]`  
Output: `0`
