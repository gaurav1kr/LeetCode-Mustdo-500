# Remove Duplicates from Sorted List II

This is an optimized C++ solution for the [LeetCode problem](https://leetcode.com/problems/remove-duplicates-from-sorted-list-ii/) "Remove Duplicates from Sorted List II".

## Problem Statement
Given the `head` of a sorted linked list, delete all nodes that have duplicate numbers, leaving only distinct numbers from the original list. Return the linked list **sorted as well**.

---

## Optimized C++ Solution

```cpp
class Solution {
public:
    ListNode* deleteDuplicates(ListNode* head) {
        ListNode dummy(0, head); // Dummy node to handle edge cases
        ListNode* prev = &dummy;

        while (head) {
            if (head->next && head->val == head->next->val) {
                while (head->next && head->val == head->next->val) 
                    head = head->next; // Skip duplicates
                prev->next = head->next; // Remove duplicates
            } else {
                prev = prev->next; // Move to next unique node
            }
            head = head->next; // Move to the next node
        }

        return dummy.next;
    }
};
```

### Explanation
1. **Dummy Node**:
   - A dummy node is introduced at the beginning of the list to handle edge cases, like when the head itself contains duplicates.
2. **Two Pointers**:
   - `prev` points to the last node before the duplicate sequence.
   - `head` traverses the list to detect and skip duplicates.
3. **Skipping Duplicates**:
   - When two consecutive nodes have the same value, move the `head` pointer until the end of the duplicate sequence is reached.
   - Update `prev->next` to skip the duplicate nodes entirely.
4. **Handling Unique Nodes**:
   - If no duplicates are found, simply advance the `prev` pointer to the next node.
5. **Result**:
   - Return `dummy.next`, which points to the modified list.

### Complexity
- **Time Complexity**: `O(n)` where `n` is the number of nodes in the list. Each node is visited once.
- **Space Complexity**: `O(1)` as no additional data structures are used.
