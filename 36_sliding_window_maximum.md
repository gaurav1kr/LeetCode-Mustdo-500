## Sliding Window Maximum - Optimal C++ Solution

### Problem Statement
Given an integer array `nums` and an integer `k`, find the maximum element in every contiguous subarray of size `k`.

### **Optimal Approach - Using Deque**
#### **Key Idea**
- Use a **deque** (double-ended queue) to store indices of array elements.
- Maintain elements in **decreasing order** in the deque.
- Remove elements that fall out of the current window.
- Remove elements from the back of the deque if they are **smaller than the current element**.
- The front of the deque holds the **maximum element index** for the current window.

---

### **C++ Code**
```cpp
#include <vector>
#include <deque>

using namespace std;

class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        vector<int> result;
        deque<int> dq; // Store indices of useful elements
        
        for (int i = 0; i < nums.size(); i++) {
            // Remove elements that are out of this window
            if (!dq.empty() && dq.front() == i - k)
                dq.pop_front();

            // Remove elements smaller than the current element (from back)
            while (!dq.empty() && nums[dq.back()] <= nums[i])
                dq.pop_back();

            // Insert current element's index
            dq.push_back(i);

            // Add the max to result (starting from the first full window)
            if (i >= k - 1)
                result.push_back(nums[dq.front()]);
        }
        
        return result;
    }
};
```

---

### **Complexity Analysis**
- **O(N) Time Complexity**: Each element is pushed and popped from the deque **at most once**.
- **O(K) Space Complexity**: The deque stores **at most k elements**.

---

### **Example Walkthrough**
#### **Input:**
```cpp
nums = [1,3,-1,-3,5,3,6,7], k = 3
```
#### **Steps:**
| Window Position | Deque State (Indices) | Max in Window |
|---------------|--------------------|---------------|
| `[1,3,-1]`   | `[3]`               | `3` |
| `[3,-1,-3]`  | `[3,-1]`            | `3` |
| `[-1,-3,5]`  | `[5]`               | `5` |
| `[-3,5,3]`   | `[5,3]`             | `5` |
| `[5,3,6]`    | `[6]`               | `6` |
| `[3,6,7]`    | `[7]`               | `7` |

#### **Output:**
```cpp
[3, 3, 5, 5, 6, 7]
```

---

### **Why This Works Efficiently?**
- **Deque always maintains the largest elements' indices.**
- **Each index is added & removed at most once → O(N) time complexity.**
- **Ensures efficient window shifting without unnecessary comparisons.**

Let me know if you need further explanation! 🚀