
# Optimized C++ Solution for LeetCode "Insert Interval"

This document provides an optimized solution for the **"Insert Interval"** problem from LeetCode.

---

## Problem Overview
- You are given a list of intervals and a new interval to insert.
- The goal is to insert the new interval into the list while maintaining the order and merging overlapping intervals.

---

## Optimized C++ Solution

```cpp
#include <vector>
using namespace std;

class Solution {
public:
    vector<vector<int>> insert(vector<vector<int>>& intervals, vector<int>& newInterval) {
        vector<vector<int>> result;
        int i = 0, n = intervals.size();

        // Step 1: Add all intervals that end before the new interval starts
        while (i < n && intervals[i][1] < newInterval[0]) {
            result.push_back(intervals[i]);
            i++;
        }

        // Step 2: Merge overlapping intervals with the new interval
        while (i < n && intervals[i][0] <= newInterval[1]) {
            newInterval[0] = min(newInterval[0], intervals[i][0]); // Update start
            newInterval[1] = max(newInterval[1], intervals[i][1]); // Update end
            i++;
        }
        result.push_back(newInterval); // Add the merged interval

        // Step 3: Add all remaining intervals
        while (i < n) {
            result.push_back(intervals[i]);
            i++;
        }

        return result;
    }
};
```

---

## Explanation
1. **Step 1:** Add all intervals that do not overlap with the new interval (i.e., intervals ending before the new interval starts).
2. **Step 2:** Merge all overlapping intervals by adjusting the `start` and `end` of the new interval as needed.
3. **Step 3:** Add any remaining intervals that start after the new interval ends.
4. The solution ensures that the final list of intervals is sorted and merged correctly.

---

## Complexity
- **Time Complexity:** \( O(n) \), where \( n \) is the number of intervals. We iterate through the list of intervals once.
- **Space Complexity:** \( O(1) \) additional space, since the result list is returned and no extra data structures are used.

---

## Example Input/Output

### Example 1:
#### Input:
```cpp
intervals = {{1, 3}, {6, 9}}
newInterval = {2, 5}
```
#### Output:
```cpp
{{1, 5}, {6, 9}}
```

### Example 2:
#### Input:
```cpp
intervals = {{1, 2}, {3, 5}, {6, 7}, {8, 10}, {12, 16}}
newInterval = {4, 8}
```
#### Output:
```cpp
{{1, 2}, {3, 10}, {12, 16}}
```

---

Feel free to use this solution and adapt it as needed for your coding interviews or practice! Let me know if you have any questions.
