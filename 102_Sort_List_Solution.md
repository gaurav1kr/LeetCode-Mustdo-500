# Solution for LeetCode Problem: Sort List

This problem asks for sorting a singly linked list in \( O(n \log n) \) time complexity and using constant space. The most efficient approach is to use the **merge sort algorithm**, which is naturally suited for linked lists.

### Optimized C++ Implementation

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
    ListNode* sortList(ListNode* head) {
        // Base case: empty list or single element list
        if (!head || !head->next) return head;

        // Split the list into two halves
        ListNode* mid = getMid(head);
        ListNode* left = head;
        ListNode* right = mid->next;
        mid->next = nullptr;

        // Recursively sort both halves
        left = sortList(left);
        right = sortList(right);

        // Merge the sorted halves
        return merge(left, right);
    }

private:
    // Function to split the list and find the middle node
    ListNode* getMid(ListNode* head) {
        ListNode* slow = head;
        ListNode* fast = head->next;

        while (fast && fast->next) {
            slow = slow->next;
            fast = fast->next->next;
        }

        return slow;
    }

    // Function to merge two sorted linked lists
    ListNode* merge(ListNode* l1, ListNode* l2) {
        ListNode dummy(0);
        ListNode* tail = &dummy;

        while (l1 && l2) {
            if (l1->val < l2->val) {
                tail->next = l1;
                l1 = l1->next;
            } else {
                tail->next = l2;
                l2 = l2->next;
            }
            tail = tail->next;
        }

        // Attach the remaining nodes, if any
        tail->next = l1 ? l1 : l2;

        return dummy.next;
    }
};
```

### Explanation

1. **Base Case**:
   - If the list is empty or contains only one node, it is already sorted, so return the head.

2. **Finding the Middle**:
   - The `getMid` function uses the two-pointer technique (slow and fast pointers) to find the middle node of the list.

3. **Recursively Divide**:
   - The list is divided into two halves: left (from head to mid) and right (from mid+1 to the end).

4. **Merge**:
   - The `merge` function merges two sorted linked lists into one sorted list.

### Complexity Analysis

- **Time Complexity**:
  - Each recursive split is \( O(\log n) \), and merging two halves is \( O(n) \), resulting in \( O(n \log n) \) overall.

- **Space Complexity**:
  - The space complexity is \( O(1) \) for merging, but \( O(\log n) \) for the recursion stack due to the recursive nature of the solution.

---

This implementation is both efficient and clean, adhering to the problem's constraints for time and space.
