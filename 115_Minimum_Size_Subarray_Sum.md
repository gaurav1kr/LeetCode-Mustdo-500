# Minimum Size Subarray Sum

## Problem Description
Given an integer array `nums` and an integer `target`, return the minimal length of a contiguous subarray of which the sum is greater than or equal to `target`. If there is no such subarray, return `0`.

---

## Optimized Solution in C++

### Code:
```cpp
#include <vector>
#include <algorithm>
#include <climits>

using namespace std;

int minSubArrayLen(int target, vector<int>& nums) {
    int n = nums.size();
    int left = 0, current_sum = 0, min_length = INT_MAX;

    for (int right = 0; right < n; ++right) {
        current_sum += nums[right];

        while (current_sum >= target) {
            min_length = min(min_length, right - left + 1);
            current_sum -= nums[left];
            ++left;
        }
    }

    return (min_length == INT_MAX) ? 0 : min_length;
}
```

### Explanation:
1. **Sliding Window Technique**:
   - Use two pointers: `left` (start of the current window) and `right` (end of the current window).
   - Expand the window by adding `nums[right]` to `current_sum`.
   - Shrink the window from the left while `current_sum >= target`, updating `min_length` as the minimal window size.

2. **Key Operations**:
   - The `while` loop ensures that we reduce the window size as much as possible when the condition `current_sum >= target` is met.
   - `min(min_length, right - left + 1)` calculates the smallest window size seen so far.

3. **Edge Cases**:
   - If no subarray meets the condition, return `0` (handled by checking if `min_length == INT_MAX` at the end).

### Complexity:
- **Time Complexity**: O(n)
  - Each element is processed at most twice (once when expanding the window and once when shrinking it).
- **Space Complexity**: O(1)
  - No additional data structures are used.

---

## Example Usage:
```cpp
#include <iostream>

int main() {
    vector<int> nums = {2, 3, 1, 2, 4, 3};
    int target = 7;
    cout << minSubArrayLen(target, nums) << endl;  // Output: 2
    return 0;
}
```

### Output:
The minimal length of a subarray with sum ≥ 7 is `2` (`[4, 3]`).

---

## Key Takeaways:
- The sliding window technique is highly efficient for problems involving contiguous subarrays.
- Always verify edge cases, such as when no valid subarray exists (e.g., target larger than the sum of all elements in the array).
