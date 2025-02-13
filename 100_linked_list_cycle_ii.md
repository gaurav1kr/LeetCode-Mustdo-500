# Linked List Cycle II - Optimized C++ Solution

## Problem Description
Given the `head` of a linked list, return the node where the cycle begins. If there is no cycle, return `nullptr`.

There is a cycle in a linked list if there is some node in the list that can be reached again by continuously following the `next` pointer. Internally, `pos` is used to denote the index of the node that tail's `next` pointer is connected to (0-indexed). It is `-1` if there is no cycle. **Note** that `pos` is not passed as a parameter.

## Approach
We use Floyd's Cycle Detection Algorithm (also known as the two-pointer approach) to efficiently detect the cycle and locate its starting node.

## Code
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
    ListNode *detectCycle(ListNode *head) {
        if (!head || !head->next) return nullptr;

        ListNode *slow = head, *fast = head;

        // Detect if a cycle exists using slow and fast pointers
        while (fast && fast->next) {
            slow = slow->next;
            fast = fast->next->next;
            if (slow == fast) break;  // Cycle detected
        }

        // If no cycle is found, return nullptr
        if (!fast || !fast->next) return nullptr;

        // Find the starting node of the cycle
        slow = head;
        while (slow != fast) {
            slow = slow->next;
            fast = fast->next;
        }

        return slow;  // Starting node of the cycle
    }
};
```

## Explanation
1. **Cycle Detection**:  
   Two pointers, `slow` and `fast`, are used. `slow` moves one step at a time, while `fast` moves two steps. If there is a cycle, the two pointers will eventually meet within the cycle.

2. **Finding the Cycle's Start Node**:  
   - Once the cycle is detected, reset one of the pointers (`slow`) to the head of the list. 
   - Move both pointers one step at a time. The node where they meet again is the start of the cycle.

3. **Edge Cases**:  
   - Empty list (`head == nullptr`) or a list with one node (`head->next == nullptr`): Return `nullptr` as no cycle exists.
   - A cycle exists in the list: Return the node where the cycle starts.

## Complexity
- **Time Complexity**: \(O(n)\), where \(n\) is the number of nodes in the list. The algorithm traverses the list at most twice.
- **Space Complexity**: \(O(1)\), as no extra space is used.

## Example
### Input:
```plaintext
head = [3,2,0,-4], pos = 1
```
### Output:
```plaintext
ListNode at position 1 (value = 2)
```

This solution is efficient and widely used for problems involving cycles in linked lists.
