
# Container With Most Water - Optimal C++ Solution

## Problem Description
The **Container With Most Water** problem on LeetCode ([link](https://leetcode.com/problems/container-with-most-water/description/)) requires finding two lines that together with the x-axis form a container such that the container holds the most water.

Given a vector of heights, you need to maximize the area between two lines while maintaining \(O(n)\) time complexity.

---

## Optimal Solution - Two-Pointer Technique
Below is an optimal C++ implementation using the two-pointer approach:

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    int maxArea(vector<int>& height) {
        int left = 0;                  // Left pointer
        int right = height.size() - 1; // Right pointer
        int maxArea = 0;               // Variable to store the maximum area

        while (left < right) {
            // Calculate the current area
            int currentArea = min(height[left], height[right]) * (right - left);
            maxArea = max(maxArea, currentArea);

            // Move the pointer pointing to the shorter line inward
            if (height[left] < height[right]) {
                ++left;
            } else {
                --right;
            }
        }

        return maxArea;
    }
};

int main() {
    vector<int> height = {1,8,6,2,5,4,8,3,7}; // Example input
    Solution solution;
    cout << "Maximum Area: " << solution.maxArea(height) << endl;
    return 0;
}
```

---

## Explanation

### Two-Pointer Approach:
1. **Initialize Two Pointers**:
   - Place one pointer (`left`) at the beginning and the other pointer (`right`) at the end of the height vector.
   
2. **Calculate Area**:
   - Compute the current area as:
     ```
     currentArea = min(height[left], height[right]) * (right - left);
     ```
   - Update the maximum area (`maxArea`) if the current area is larger.

3. **Move the Pointer**:
   - Move the pointer pointing to the shorter line inward (i.e., increment `left` or decrement `right`) to maximize the potential area.

4. **Stop Condition**:
   - When `left` meets `right`, the search ends.

### Complexity Analysis:
- **Time Complexity**: \(O(n)\)
  - Each element is processed at most once since the pointers move inward.
- **Space Complexity**: \(O(1)\)
  - No additional space is required beyond a few variables.

---

## Example Input and Output
### Input:
```plaintext
height = [1,8,6,2,5,4,8,3,7]
```
### Output:
```plaintext
Maximum Area: 49
```

---

## Key Takeaways
- The two-pointer approach efficiently reduces the problem's complexity to \(O(n)\) by eliminating unnecessary calculations.
- Always move the pointer at the shorter line inward to maximize the area potential.
