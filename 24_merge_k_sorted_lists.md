# Merge K Sorted Lists - Optimized C++ Solution

## Problem Statement
Given an array of `k` linked lists, each sorted in ascending order, merge all the linked lists into one sorted linked list and return it.

## Optimized C++ Solution

```cpp
#include <queue>
using namespace std;

struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(nullptr) {}
};

struct Compare {
    bool operator()(ListNode* a, ListNode* b) {
        return a->val > b->val;
    }
};

ListNode* mergeKLists(vector<ListNode*>& lists) {
    priority_queue<ListNode*, vector<ListNode*>, Compare> pq;
    
    for (auto list : lists)
        if (list) pq.push(list);
    
    ListNode dummy(0), *tail = &dummy;
    
    while (!pq.empty()) {
        tail->next = pq.top();
        pq.pop();
        tail = tail->next;
        if (tail->next) pq.push(tail->next);
    }
    
    return dummy.next;
}
```

## Explanation
1. **Min-Heap (Priority Queue):**  
   - Stores the heads of all linked lists.
   - Always pops the smallest element and inserts the next node from the same list.
   
2. **Heap Operations:**  
   - Insertion takes **O(log k)**.
   - Extracting the minimum element takes **O(log k)**.
   - Since each of the **N** nodes is processed once, the total time complexity is **O(N log k)**.

## Complexity Analysis
- **Time Complexity:** `O(N log k)`, where `N` is the total number of nodes and `k` is the number of linked lists.
- **Space Complexity:** `O(k)`, since the priority queue holds at most `k` elements.

This is an **optimal and efficient** approach for solving the problem.
