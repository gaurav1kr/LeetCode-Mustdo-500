# Finding the Duplicate Number - Floyd's Tortoise and Hare Algorithm

## Problem Statement
Given an array of integers `nums` containing `n + 1` integers where each integer is in the range `[1, n]` inclusive, there is only **one repeated number** in `nums`. Find and return this duplicate number.

### Constraints:
- `1 <= n <= 10^5`
- `nums.length == n + 1`
- Each integer appears at least once.
- Only one duplicate number exists.
- The array cannot be modified.

---

## Optimal Solution: Floyd's Tortoise and Hare (Cycle Detection)
This problem can be efficiently solved using **Floyd's Cycle Detection Algorithm**, also known as the **Tortoise and Hare Algorithm**.

### Explanation:
1. **Think of the array as a Linked List:**
   - Each index represents a node.
   - Each value in the array represents the next pointer.
   - Since there are `n + 1` numbers in the range `[1, n]`, a cycle must exist.

2. **Finding the Cycle:**
   - Use two pointers: `slow` and `fast`.
   - Move `slow` one step at a time (`slow = nums[slow]`).
   - Move `fast` two steps at a time (`fast = nums[nums[fast]]`).
   - When `slow == fast`, a cycle is detected.

3. **Finding the Duplicate Number:**
   - Reset `slow` to the start of the array (`nums[0]`).
   - Move both `slow` and `fast` **one step at a time**.
   - The meeting point is the duplicate number.

### Code Implementation (C++):
```cpp
class Solution {
public:
    int findDuplicate(vector<int>& nums) {
        int slow = nums[0], fast = nums[0];
        
        // Phase 1: Detect cycle
        do {
            slow = nums[slow];
            fast = nums[nums[fast]];
        } while (slow != fast);
        
        // Phase 2: Find the entrance to the cycle
        slow = nums[0];
        while (slow != fast) {
            slow = nums[slow];
            fast = nums[fast];
        }
        return slow;
    }
};
```

### Complexity Analysis:
- **Time Complexity:** `O(n)`, as we traverse the array at most twice.
- **Space Complexity:** `O(1)`, as only two extra variables (`slow` and `fast`) are used.

### Why This Works:
- Since numbers act as pointers, they form a **cycle**.
- The duplicate number is the **entry point** of this cycle.
- Floyd's algorithm efficiently finds this entry point.

---

## Alternative Approaches
| Approach | Time Complexity | Space Complexity | Modifies Array? |
|----------|---------------|----------------|---------------|
| Sorting | O(n log n) | O(1) | Yes |
| HashSet | O(n) | O(n) | No |
| Binary Search (on values) | O(n log n) | O(1) | No |
| Bitwise Counting | O(n log n) | O(1) | No |
| **Floyd’s Cycle Detection** | **O(n)** | **O(1)** | **No** |

Floyd’s Cycle Detection is the most efficient and recommended solution.

---

## Conclusion
Floyd’s Tortoise and Hare algorithm is an optimal and clever technique for detecting cycles in linked lists or sequences mapped like linked lists. It works efficiently in `O(n)` time and `O(1)` space, making it the best approach to solve the **Find the Duplicate Number** problem without modifying the array. 🚀
