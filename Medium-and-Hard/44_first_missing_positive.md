## First Missing Positive - LeetCode Solution

### Approach:
We solve the problem using an **in-place cyclic sort** technique, which ensures an **O(n) time complexity** with **O(1) extra space**.

### Algorithm:
1. Iterate through the array and place each number in its correct position if it lies in the range `[1, n]`.
2. Traverse the array again to find the first index where `nums[i] != i + 1`.
3. If all numbers are correctly placed, return `n + 1`.

### C++ Code:
```cpp
#include <vector>
using namespace std;

class Solution {
public:
    int firstMissingPositive(vector<int>& nums) {
        int n = nums.size();
        
        // Place numbers in their correct positions
        for (int i = 0; i < n; i++) {
            while (nums[i] > 0 && nums[i] <= n && nums[i] != nums[nums[i] - 1]) {
                swap(nums[i], nums[nums[i] - 1]);
            }
        }
        
        // Identify the first missing positive
        for (int i = 0; i < n; i++) {
            if (nums[i] != i + 1) {
                return i + 1;
            }
        }
        
        return n + 1;
    }
};
```

### Complexity Analysis:
- **Time Complexity:** `O(n)` (Each number is swapped at most once)
- **Space Complexity:** `O(1)` (In-place sorting)

### Example:
#### **Input:**
```cpp
nums = [3, 4, -1, 1]
```
#### **Processing:**
After placement:
```cpp
nums = [1, -1, 3, 4]
```
First missing positive: **2**

#### **Output:**
```cpp
2
```