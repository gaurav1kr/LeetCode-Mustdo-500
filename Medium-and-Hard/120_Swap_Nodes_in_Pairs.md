
# Swap Nodes in Pairs

## Problem Statement
Given a linked list, swap every two adjacent nodes and return its head. You must solve the problem without modifying the values in the list's nodes (i.e., only nodes themselves may be changed.)

## Solution

### C++ Implementation
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
    ListNode* swapPairs(ListNode* head) {
        if (!head || !head->next) return head; // Base case: If list has 0 or 1 node, no swapping needed.
        
        ListNode* newHead = head->next; // The new head will be the second node.
        ListNode* prev = nullptr;
        while (head && head->next) {
            ListNode* first = head;
            ListNode* second = head->next;
            
            // Swap the pair
            first->next = second->next;
            second->next = first;
            if (prev) prev->next = second; // Connect the previous pair to the current swapped pair.
            
            // Move pointers forward
            prev = first;
            head = first->next;
        }
        return newHead;
    }
};
```

### Explanation
1. **Base Case**: If the list is empty or has only one node, return the head as-is.
2. **Two-pointer Technique**:
   - `first`: Points to the first node of the pair.
   - `second`: Points to the second node of the pair.
3. **Swapping Logic**:
   - Adjust the `next` pointers of the two nodes in the pair to swap them.
   - Connect the previous pair's last node (`prev`) to the new pair's first node (`second`).
4. **Iterate**:
   - Move the `head` and `prev` pointers to the next pair of nodes for further processing.
5. **New Head**:
   - The second node of the first pair becomes the new head of the swapped list.

### Complexity Analysis
- **Time Complexity**: \(O(n)\), where \(n\) is the number of nodes in the list. We traverse the list once.
- **Space Complexity**: \(O(1)\), as no extra space is used.

### Example
#### Input:
`head = [1, 2, 3, 4]`

#### Output:
`[2, 1, 4, 3]`

---

This implementation efficiently solves the problem while adhering to the constraints. Let me know if you have any questions or need further assistance!
