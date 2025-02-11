# Odd-Even Linked List Solution

Here is an optimized and concise C++ solution for the **Odd-Even Linked List** problem:

```cpp
class Solution {
public:
    ListNode* oddEvenList(ListNode* head) {
        if (!head || !head->next) return head; // Handle edge cases

        ListNode *odd = head, *even = head->next, *evenHead = even;
        while (even && even->next) {
            odd->next = even->next;
            odd = odd->next;
            even->next = odd->next;
            even = even->next;
        }
        odd->next = evenHead; // Connect odd list to even list
        return head;
    }
};
```

## Explanation:
1. **Edge Case Check:**
   - If the list has 0 or 1 nodes, return the head as no rearrangement is needed.

2. **Pointers Setup:**
   - Use two pointers (`odd` and `even`) to traverse the odd and even nodes separately.
   - Maintain the head of the even list (`evenHead`) for later connection.

3. **Rearrangement:**
   - Iterate through the list and alternate the odd and even connections.

4. **Final Connection:**
   - Attach the end of the odd list to the head of the even list.

## Complexity:
- **Time Complexity:** `O(n)` - where `n` is the number of nodes in the linked list, as we iterate through the list once.
- **Space Complexity:** `O(1)` - no additional space is used.

This solution is efficient and meets the problem's constraints. Let me know if you have any questions or need further assistance!
