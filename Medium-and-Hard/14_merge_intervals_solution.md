# Optimized C++ Solution for Merge Intervals

Below is a concise and efficient solution for the [Merge Intervals](https://leetcode.com/problems/merge-intervals/) problem:

```cpp
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    vector<vector<int>> merge(vector<vector<int>>& intervals) {
        if (intervals.empty()) return {};
        
        // Sort intervals by the starting point
        sort(intervals.begin(), intervals.end());
        
        vector<vector<int>> merged;
        
        for (const auto& interval : intervals) {
            // If merged is empty or the current interval does not overlap, add it to the result
            if (merged.empty() || merged.back()[1] < interval[0]) {
                merged.push_back(interval);
            } else {
                // Merge intervals by updating the end of the last interval in merged
                merged.back()[1] = max(merged.back()[1], interval[1]);
            }
        }
        
        return merged;
    }
};
```

## Explanation
1. **Sorting**:
   - The intervals are sorted by their starting points to ensure proper order for processing.

2. **Merging Logic**:
   - If the current interval overlaps with the last interval in the result, their endpoints are merged by taking the maximum.
   - Otherwise, the current interval is added to the result as-is.

3. **Edge Case**:
   - If the input list is empty, the function immediately returns an empty list.

## Complexity Analysis
- **Time Complexity**: 
  - Sorting the intervals takes \(O(n \log n)\), where \(n\) is the number of intervals.
  - The single pass through the intervals for merging is \(O(n)\).
  - Overall: \(O(n \log n)\).

- **Space Complexity**: 
  - \(O(n)\) for the output list of merged intervals.

This solution is both time and space efficient, making it suitable for handling large input sizes.
