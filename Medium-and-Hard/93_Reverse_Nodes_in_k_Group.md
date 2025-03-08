# Reverse Nodes in k-Group - Solution

## Problem Description
Given a linked list, reverse the nodes of the list `k` at a time and return its modified list. `k` is a positive integer and is less than or equal to the length of the linked list. If the number of nodes is not a multiple of `k`, leave the last nodes as they are.

### Constraints
- The number of nodes in the list is `n`.
- \(1 \leq k \leq n \leq 5000\)
- \(0 \leq Node.val \leq 1000\)

---

## Optimized C++ Solution
```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* reverseKGroup(ListNode* head, int k) {
        if (!head || k == 1) return head;

        // Dummy node
        ListNode dummy(0);
        dummy.next = head;
        ListNode* prev = &dummy, *curr = head, *next = nullptr;

        // Count total number of nodes
        int count = 0;
        while (curr) {
            count++;
            curr = curr->next;
        }

        // Reverse nodes in k groups
        curr = head;
        while (count >= k) {
            next = curr->next;
            for (int i = 1; i < k; i++) {
                curr->next = next->next;
                next->next = prev->next;
                prev->next = next;
                next = curr->next;
            }
            prev = curr;
            curr = next;
            count -= k;
        }

        return dummy.next;
    }
};
```

---

## Explanation
1. **Base Case**: If the list is empty or `k == 1`, no reversing is needed.
2. **Dummy Node**: Introduced for easy manipulation of the head pointer.
3. **Counting Nodes**: Counts the total number of nodes to determine how many full `k` groups exist.
4. **Reversing in Groups**:
   - Use three pointers (`prev`, `curr`, and `next`) to reverse nodes in-place.
   - Perform `k-1` pointer adjustments for each group to reverse the group.
5. **End Handling**: If the remaining nodes are less than `k`, they are left as-is.

---

## Complexity Analysis
- **Time Complexity**: \(O(N)\), where \(N\) is the number of nodes in the linked list (each node is processed exactly once).
- **Space Complexity**: \(O(1)\), as the reversal is done in-place.

---

## Example
### Input
```
head = [1,2,3,4,5], k = 2
```
### Output
```
[2,1,4,3,5]
```
### Input
```
head = [1,2,3,4,5], k = 3
```
### Output
```
[3,2,1,4,5]
```

This solution is both clean and efficient, making it ideal for competitive programming or interviews.
