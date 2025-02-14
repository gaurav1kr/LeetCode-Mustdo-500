
# Solution for LeetCode Problem: Add Two Numbers

This is an optimal C++ solution for the [Add Two Numbers](https://leetcode.com/problems/add-two-numbers/) problem. The solution uses efficient linked list traversal and handles carry propagation gracefully.

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
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        ListNode dummy(0); // Dummy node to simplify edge cases
        ListNode* current = &dummy;
        int carry = 0;

        while (l1 || l2 || carry) {
            int sum = carry;
            if (l1) {
                sum += l1->val;
                l1 = l1->next;
            }
            if (l2) {
                sum += l2->val;
                l2 = l2->next;
            }

            carry = sum / 10;
            current->next = new ListNode(sum % 10);
            current = current->next;
        }

        return dummy.next;
    }
};
```

## Key Features of the Code
1. **Dummy Node**: A dummy node simplifies edge cases, especially for initializing the linked list.
2. **Carry Management**: Handles cases where a carry extends the length of the result list.
3. **Iterative Traversal**: Traverses both lists simultaneously until both are exhausted.

## Complexity Analysis
- **Time Complexity**: \(O(\max(m, n))\), where \(m\) and \(n\) are the lengths of `l1` and `l2`. Each node is visited once.
- **Space Complexity**: \(O(\max(m, n))\), due to the space required for the result linked list.

## Example Usage
Given two non-empty linked lists representing two non-negative integers:

- Input:
  - \(l1 = [2, 4, 3]\)
  - \(l2 = [5, 6, 4]\)

- Output:
  - \([7, 0, 8]\) (Explanation: 342 + 465 = 807)

## Notes
- The provided code assumes well-formed input linked lists.
- Make sure to free the allocated memory for the result linked list in production environments.
