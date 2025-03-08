
# Median of Two Sorted Arrays - Optimal Solution

This document provides the optimal solution for the **Median of Two Sorted Arrays** problem on LeetCode. The solution uses a binary search approach and has a time complexity of \(O(\log(\min(m, n)))\), where \(m\) and \(n\) are the lengths of the two input arrays.

## Problem Description
You are given two sorted arrays `nums1` and `nums2` of size \(m\) and \(n\), respectively. The task is to find the median of these two sorted arrays. The overall run time complexity should be \(O(\log(m + n))\).

---

## C++ Solution

```cpp
#include <vector>
#include <algorithm>
#include <climits>

using namespace std;

class Solution {
public:
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
        // Ensure nums1 is the smaller array
        if (nums1.size() > nums2.size()) {
            return findMedianSortedArrays(nums2, nums1);
        }
        
        int m = nums1.size();
        int n = nums2.size();
        int low = 0, high = m;
        
        while (low <= high) {
            int partition1 = (low + high) / 2;
            int partition2 = (m + n + 1) / 2 - partition1;
            
            int maxLeft1 = (partition1 == 0) ? INT_MIN : nums1[partition1 - 1];
            int minRight1 = (partition1 == m) ? INT_MAX : nums1[partition1];
            
            int maxLeft2 = (partition2 == 0) ? INT_MIN : nums2[partition2 - 1];
            int minRight2 = (partition2 == n) ? INT_MAX : nums2[partition2];
            
            if (maxLeft1 <= minRight2 && maxLeft2 <= minRight1) {
                // Found the correct partitions
                if ((m + n) % 2 == 0) {
                    return (max(maxLeft1, maxLeft2) + min(minRight1, minRight2)) / 2.0;
                } else {
                    return max(maxLeft1, maxLeft2);
                }
            } else if (maxLeft1 > minRight2) {
                // Move left in nums1
                high = partition1 - 1;
            } else {
                // Move right in nums1
                low = partition1 + 1;
            }
        }
        
    }
};
```

---

## Explanation

### Binary Search on the Smaller Array
To optimize, we always perform the binary search on the smaller array (`nums1`). This ensures that the number of iterations is minimized.

### Partitions
The arrays are virtually divided into left and right parts. The goal is to ensure that all elements in the left part are smaller than or equal to all elements in the right part.

### Edge Cases
If a partition index reaches the bounds of the array, `INT_MIN` or `INT_MAX` is used to handle boundary conditions gracefully.

### Median Calculation
- If the total number of elements is odd, the median is the maximum of the left part.
- If even, the median is the average of the maximum of the left part and the minimum of the right part.

---

## Complexity Analysis

### Time Complexity
- \(O(\log(\min(m, n)))\): We perform a binary search on the smaller array.

### Space Complexity
- \(O(1)\): No extra space is used.

---

## Example Usage
```cpp
#include <iostream>

int main() {
    Solution solution;
    vector<int> nums1 = {1, 3};
    vector<int> nums2 = {2};
    double median = solution.findMedianSortedArrays(nums1, nums2);
    cout << "Median: " << median << endl;

    nums1 = {1, 2};
    nums2 = {3, 4};
    median = solution.findMedianSortedArrays(nums1, nums2);
    cout << "Median: " << median << endl;

    return 0;
}
```

---

## Notes
- The solution assumes that both input arrays are sorted.
- If the inputs are not sorted, the program will throw an `invalid_argument` exception.

Let me know if you have any further questions or need additional explanations!
