## House Robber - Optimal C++ Solution

### Problem Description
The **House Robber** problem on LeetCode requires determining the maximum amount of money a robber can steal without robbing two adjacent houses.

[LeetCode Problem Link](https://leetcode.com/problems/house-robber/description/)

---

### **Optimal Approach**
The problem can be solved efficiently using **Dynamic Programming (DP)** with space optimization.

#### **Approach:**
1. Maintain two variables:
   - `prev1`: Stores the maximum amount robbed up to the previous house.
   - `prev2`: Stores the maximum amount robbed up to the house before the previous house.
2. Iterate through the `nums` array and decide whether to rob the current house or not.
3. The final result is stored in `prev1`.

---

### **Code Implementation:**
```cpp
#include <vector>
using namespace std;

class Solution {
public:
    int rob(vector<int>& nums) {
        int prev1 = 0, prev2 = 0;
        
        for (int num : nums) {
            int temp = prev1;
            prev1 = max(prev2 + num, prev1);
            prev2 = temp;
        }
        
        return prev1;
    }
};
```

---

### **Complexity Analysis:**
- **Time Complexity:** \( O(n) \) (since we iterate through the array once).
- **Space Complexity:** \( O(1) \) (as only two extra variables are used instead of a DP array).

---

### **Explanation with Example:**
#### **Input:**
```cpp
nums = [2,7,9,3,1]
```
#### **Iterations:**
| House | Amount | Max if Robbed | Max if Skipped | Current Max |
|--------|--------|--------------|--------------|--------------|
| 1st    | 2      | 2            | 0            | 2            |
| 2nd    | 7      | 7            | 2            | 7            |
| 3rd    | 9      | 2 + 9 = 11   | 7            | 11           |
| 4th    | 3      | 7 + 3 = 10   | 11           | 11           |
| 5th    | 1      | 11 + 1 = 12  | 11           | 12           |

#### **Output:**
```cpp
12
```

---

### **Key Takeaways:**
- We use two variables instead of an entire DP array, reducing space complexity to \( O(1) \).
- The approach efficiently computes the maximum loot possible while ensuring no two adjacent houses are robbed.
- This is the **most optimized** approach in terms of both **time** and **space complexity**.

🚀 **Happy Coding!**
