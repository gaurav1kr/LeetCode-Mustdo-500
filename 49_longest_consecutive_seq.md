## Longest Consecutive Sequence - LeetCode Solution

### **Problem Statement:**
Given an unsorted array of integers `nums`, return the length of the longest consecutive elements sequence.

**Example:**
```cpp
Input: nums = [100, 4, 200, 1, 3, 2]
Output: 4
Explanation: The longest consecutive sequence is [1, 2, 3, 4].
```

### **Approach:**
1. Use an **unordered_set** to store all unique numbers.
2. Iterate through each number in the set and check **only for the start** of a sequence (i.e., a number without a preceding number in the set).
3. Expand the sequence by checking consecutive numbers.
4. Keep track of the longest sequence encountered.

### **C++ Code:**
```cpp
#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int longestConsecutive(vector<int>& nums) {
        unordered_set<int> numSet(nums.begin(), nums.end());
        int longestStreak = 0;

        for (int num : numSet) {
            // Only check for the start of a sequence
            if (numSet.find(num - 1) == numSet.end()) {
                int currentNum = num;
                int currentStreak = 1;

                // Expand the sequence
                while (numSet.find(currentNum + 1) != numSet.end()) {
                    currentNum++;
                    currentStreak++;
                }

                longestStreak = max(longestStreak, currentStreak);
            }
        }
        return longestStreak;
    }
};
```

### **Complexity Analysis:**
- **Time Complexity:** `O(n)`, since each number is processed only once.
- **Space Complexity:** `O(n)`, for storing numbers in an **unordered_set**.

### **Why is this Optimal?**
- **Brute force sorting takes `O(n log n)`, but this approach runs in `O(n)`.**
- **Only checks numbers that could be the start of a sequence.**
- **Uses HashSet for fast lookups instead of nested loops.**
