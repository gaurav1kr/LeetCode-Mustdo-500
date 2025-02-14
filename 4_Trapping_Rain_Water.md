
# Trapping Rain Water Problem - Optimal C++ Solution

The Trapping Rain Water problem on LeetCode can be solved optimally using a two-pointer approach. Below is the explanation and the corresponding code.

## Problem Link
[Trapping Rain Water - LeetCode](https://leetcode.com/problems/trapping-rain-water/description/)

## Optimal C++ Solution
```cpp
#include <vector>
#include <algorithm>
using namespace std;

class Solution {
public:
    int trap(vector<int>& height) {
        int left = 0, right = height.size() - 1;
        int leftMax = 0, rightMax = 0;
        int waterTrapped = 0;

        while (left < right) {
            if (height[left] < height[right]) {
                if (height[left] >= leftMax) {
                    leftMax = height[left];
                } else {
                    waterTrapped += leftMax - height[left];
                }
                left++;
            } else {
                if (height[right] >= rightMax) {
                    rightMax = height[right];
                } else {
                    waterTrapped += rightMax - height[right];
                }
                right--;
            }
        }

        return waterTrapped;
    }
};
```

## Explanation
1. **Two Pointers:**  
   - Use two pointers, `left` and `right`, to traverse the height array from both ends.
   - `leftMax` and `rightMax` are used to store the maximum heights encountered so far from the left and right ends.

2. **Logic:**  
   - If `height[left] < height[right]`, calculate the water trapped on the left pointer and move it inward.
   - If `height[left] >= height[right]`, calculate the water trapped on the right pointer and move it inward.
   - The key is to compare the height at the pointers with their respective `leftMax` and `rightMax`.

3. **Complexity:**  
   - **Time Complexity:** \(O(n)\), as the array is traversed only once.
   - **Space Complexity:** \(O(1)\), as no additional space is used apart from a few variables.

## Example Usage
```cpp
#include <iostream>
#include <vector>
using namespace std;

int main() {
    Solution solution;
    vector<int> height = {0,1,0,2,1,0,1,3,2,1,2,1};
    cout << "Water trapped: " << solution.trap(height) << endl;
    return 0;
}
```

### Input:
```plaintext
height = [0,1,0,2,1,0,1,3,2,1,2,1]
```
### Output:
```plaintext
Water trapped: 6
```
