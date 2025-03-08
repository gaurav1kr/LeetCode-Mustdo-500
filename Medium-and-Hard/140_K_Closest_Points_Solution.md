
# Solution for LeetCode Problem: K Closest Points to Origin

## Problem Description
Given an array of points where `points[i] = [xi, yi]` represents a point on the X-Y plane and an integer `k`, return the `k` closest points to the origin `(0, 0)`.

The distance between two points on the X-Y plane is calculated as:

\[
\text{Distance} = \sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2}
\]

You may return the answer in any order. The answer is guaranteed to be unique (except for the order).

---

## Optimized C++ Solution
Here is an optimized solution using a max-heap:

```cpp
#include <vector>
#include <queue>
#include <cmath>

using namespace std;

class Solution {
public:
    vector<vector<int>> kClosest(vector<vector<int>>& points, int k) {
        // Max-heap to store the k closest points
        priority_queue<pair<int, vector<int>>> maxHeap;

        for (const auto& point : points) {
            int x = point[0], y = point[1];
            int dist = x * x + y * y; // Compute squared distance to avoid floating-point operations
            maxHeap.push({dist, point});

            // Maintain size of the heap to at most k
            if (maxHeap.size() > k) {
                maxHeap.pop();
            }
        }

        vector<vector<int>> result;
        while (!maxHeap.empty()) {
            result.push_back(maxHeap.top().second);
            maxHeap.pop();
        }
        return result;
    }
};
```

---

## Explanation

1. **Distance Calculation**:
   - Instead of calculating the Euclidean distance (which involves a square root), we use the squared distance (`x² + y²`). This avoids unnecessary floating-point calculations since we only need relative distances.

2. **Priority Queue**:
   - A max-heap (`priority_queue`) is used to keep track of the closest `k` points. The heap stores pairs of `{distance, point}`.

3. **Heap Maintenance**:
   - If the size of the heap exceeds `k`, the farthest point (at the top of the max-heap) is removed.

4. **Complexity**:
   - **Time Complexity**: \(O(n \log k)\), where \(n\) is the number of points. Each point is pushed and popped from the heap at most once, and heap operations take \(O(\log k)\).
   - **Space Complexity**: \(O(k)\), for the heap.

5. **Output**:
   - After processing all points, the remaining elements in the heap are the `k` closest points, which are extracted into the result vector.

---

## Example

### Input:
```plaintext
points = [[1,3],[-2,2]], k = 1
```

### Output:
```plaintext
[[-2,2]]
```

### Explanation:
The distance of (1, 3) from the origin is \(\sqrt{10}\).
The distance of (-2, 2) from the origin is \(\sqrt{8}\).
Since \(\sqrt{8} < \sqrt{10}\), the closest point is `[-2, 2]`.

---

## Notes
This solution is efficient and avoids unnecessary computations, making it suitable for handling large inputs. Let me know if you have further questions or need additional variations of the solution.
