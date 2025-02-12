
# Reorder List - LeetCode Problem Solution

This Markdown file contains an optimized and concise C++ solution for the [Reorder List](https://leetcode.com/problems/reorder-list) problem on LeetCode.

---

## Problem Description

Reorder the linked list by alternating nodes from the beginning and end of the list.

### Steps to Solve:

1. **Find the Middle**: Use the slow and fast pointer technique to locate the middle of the linked list.
2. **Reverse the Second Half**: Reverse the second half of the linked list starting from the middle node.
3. **Merge the Lists**: Alternately merge nodes from the first and reversed second halves.

---

## C++ Solution

```cpp
#include <iostream>
using namespace std;

// Definition for singly-linked list.
struct ListNode {
    int val;
    ListNode* next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode* next) : val(x), next(next) {}
};

class Solution {
public:
    void reorderList(ListNode* head) {
        if (!head || !head->next || !head->next->next) return;

        // Step 1: Find the middle of the list
        ListNode* slow = head, *fast = head;
        while (fast && fast->next) {
            slow = slow->next;
            fast = fast->next->next;
        }

        // Step 2: Reverse the second half of the list
        ListNode* prev = nullptr, *curr = slow, *next = nullptr;
        while (curr) {
            next = curr->next;
            curr->next = prev;
            prev = curr;
            curr = next;
        }

        // Step 3: Merge the two halves
        ListNode* first = head, *second = prev;
        while (second->next) {
            ListNode* temp1 = first->next;
            ListNode* temp2 = second->next;
            first->next = second;
            second->next = temp1;
            first = temp1;
            second = temp2;
        }
    }
};
```

---

### Complexity Analysis

- **Time Complexity**: \(O(n)\)
    - Finding the middle: \(O(n)\)
    - Reversing the list: \(O(n)\)
    - Merging the lists: \(O(n)\)
- **Space Complexity**: \(O(1)\), as the solution modifies the list in place.

---

This solution is efficient and modifies the list in-place with no additional data structures.
