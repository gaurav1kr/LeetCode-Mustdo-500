# Reverse Linked List II - Optimized C++ Solution

This markdown file provides an optimized and concise solution for the **[Reverse Linked List II](https://leetcode.com/problems/reverse-linked-list-ii/description/)** problem on LeetCode.

## Problem Statement
You are given the `head` of a singly linked list and two integers `left` and `right` where `left <= right`. Reverse the nodes of the list from position `left` to position `right`, and return the reversed list.

## C++ Solution
Here is an optimized and clean C++ solution:

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
    ListNode* reverseBetween(ListNode* head, int left, int right) {
        if (!head || left == right) return head;

        ListNode dummy(0);
        dummy.next = head;
        ListNode* prev = &dummy;

        // Move `prev` to the node before `left`
        for (int i = 1; i < left; ++i) {
            prev = prev->next;
        }

        // Reverse the sublist
        ListNode* curr = prev->next;
        ListNode* next = nullptr;
        for (int i = 0; i < right - left; ++i) {
            next = curr->next;
            curr->next = next->next;
            next->next = prev->next;
            prev->next = next;
        }

        return dummy.next;
    }
};
```

## Explanation
1. **Dummy Node**: A dummy node is used to simplify edge cases, such as reversing the head of the list.
2. **Pointer Movement**:
   - Move the `prev` pointer to the node immediately before the `left` position.
3. **Sublist Reversal**:
   - Reverse the sublist from position `left` to `right` using an in-place reversal technique.
   - Use three pointers (`curr`, `next`, and `prev`) to rearrange the links.
4. **Return Result**:
   - The `dummy.next` pointer holds the head of the modified list.

## Complexity
- **Time Complexity**: `O(N)` where `N` is the number of nodes in the list. The solution makes a single pass through the relevant portion of the list.
- **Space Complexity**: `O(1)` as the reversal is done in-place without using extra space.

## Example
### Input:
```text
head = [1, 2, 3, 4, 5], left = 2, right = 4
```
### Output:
```text
[1, 4, 3, 2, 5]
```

## Notes
- The solution is efficient and operates directly on the list without using additional data structures.
- Ensure edge cases (e.g., `left == right`, single-node lists) are tested.
