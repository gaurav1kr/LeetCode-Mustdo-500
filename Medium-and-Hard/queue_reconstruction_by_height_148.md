
# Queue Reconstruction by Height

This document explains the solution for the [Queue Reconstruction by Height](https://leetcode.com/problems/queue-reconstruction-by-height/) problem on LeetCode.

## Problem Description
You are given an array of people, where `people[i] = [h_i, k_i]`:
- `h_i` is the height of the `i`-th person.
- `k_i` is the number of people in front of the `i`-th person who have a height greater than or equal to `h_i`.

Reconstruct and return the queue that is consistent with the given parameters.

### Example
**Input:**
```cpp
[[7, 0], [4, 4], [7, 1], [5, 0], [6, 1], [5, 2]]
```

**Output:**
```cpp
[[5, 0], [7, 0], [5, 2], [6, 1], [4, 4], [7, 1]]
```

## Optimized C++ Solution
Below is the optimized C++ code for solving this problem:

```cpp
#include <vector>
#include <algorithm>
using namespace std;

class Solution {
public:
    vector<vector<int>> reconstructQueue(vector<vector<int>>& people) {
        // Sort by height in descending order, and by k in ascending order if heights are equal
        sort(people.begin(), people.end(), [](const vector<int>& a, const vector<int>& b) {
            return a[0] > b[0] || (a[0] == b[0] && a[1] < b[1]);
        });

        vector<vector<int>> result;
        // Insert each person at the index specified by their k value
        for (const auto& person : people) {
            result.insert(result.begin() + person[1], person);
        }

        return result;
    }
};
```

## Explanation

1. **Sorting:**
   - Sort the `people` array in descending order of height (`a[0] > b[0]`).
   - If two people have the same height, sort them by ascending `k` value (`a[1] < b[1]`).

2. **Reconstruction:**
   - Use a greedy approach. Iterate through the sorted list and insert each person into the `result` vector at the index specified by their `k` value.
   - This approach ensures taller people are placed first, and the insertion of shorter people doesn’t disrupt the relative order.

## Complexity Analysis
- **Sorting:** \(O(n \log n)\), where \(n\) is the number of people.
- **Insertion:** \(O(n^2)\) in the worst case (vector insertion at a specific position takes \(O(n)\)).
- **Overall:** \(O(n^2)\).

## Example Walkthrough

### Input:
```cpp
[[7, 0], [4, 4], [7, 1], [5, 0], [6, 1], [5, 2]]
```

### Steps:
1. **After Sorting:**
   ```cpp
   [[7, 0], [7, 1], [6, 1], [5, 0], [5, 2], [4, 4]]
   ```

2. **Reconstruction:**
   - Insert `[7, 0]`: `[[7, 0]]`
   - Insert `[7, 1]`: `[[7, 0], [7, 1]]`
   - Insert `[6, 1]`: `[[7, 0], [6, 1], [7, 1]]`
   - Insert `[5, 0]`: `[[5, 0], [7, 0], [6, 1], [7, 1]]`
   - Insert `[5, 2]`: `[[5, 0], [7, 0], [5, 2], [6, 1], [7, 1]]`
   - Insert `[4, 4]`: `[[5, 0], [7, 0], [5, 2], [6, 1], [4, 4], [7, 1]]`

### Output:
```cpp
[[5, 0], [7, 0], [5, 2], [6, 1], [4, 4], [7, 1]]
```

## Key Insights
- Sorting simplifies the reconstruction as taller people are handled first.
- Inserting people by their `k` values directly maintains the constraints efficiently.

This solution is both intuitive and optimal for this problem.
