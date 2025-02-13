
# Solution to "Copy List with Random Pointer" on LeetCode

## Problem Description
You are given a linked list where each node contains an additional random pointer that could point to any node in the list or `null`. Construct a deep copy of the list. The deep copy should consist of exactly `n` brand new nodes, where each new node has its value set to the value of its corresponding original node. Both the `next` and `random` pointer of the new nodes should point to new nodes in the copied list such that the pointers in the original list and copied list represent the same list state. None of the pointers in the new list should point to nodes in the original list.

## Optimal Solution
Below is the C++ solution using a hash map for mapping original nodes to their corresponding new nodes.

### C++ Code
```cpp
#include <unordered_map>

// Definition for a Node.
class Node {
public:
    int val;
    Node* next;
    Node* random;

    Node(int _val) {
        val = _val;
        next = nullptr;
        random = nullptr;
    }
};

class Solution {
public:
    Node* copyRandomList(Node* head) {
        if (!head) return nullptr;

        // Step 1: Create a mapping from original nodes to new nodes
        std::unordered_map<Node*, Node*> nodeMap;
        Node* current = head;

        // Create all the new nodes and store in the map
        while (current) {
            nodeMap[current] = new Node(current->val);
            current = current->next;
        }

        // Step 2: Set next and random pointers for the new nodes
        current = head;
        while (current) {
            nodeMap[current]->next = nodeMap[current->next];  // Set next pointer
            nodeMap[current]->random = nodeMap[current->random];  // Set random pointer
            current = current->next;
        }

        // Return the head of the new list
        return nodeMap[head];
    }
};
```

## Explanation
### Step 1: Create New Nodes
- Use an unordered map (`std::unordered_map`) to map each node in the original list to its corresponding new node.
- Iterate through the original list and create a new node for each node in the original list. Store these new nodes in the map with the original nodes as keys.

### Step 2: Set `next` and `random` Pointers
- Iterate through the original list again.
- For each node in the original list, use the map to set the `next` and `random` pointers for the corresponding new node.

### Step 3: Return the New List
- Return the head of the new list, which is stored in `nodeMap[head]`.

## Complexity
### Time Complexity
- **O(N):**
  - The first pass creates all the new nodes (O(N)).
  - The second pass sets the `next` and `random` pointers (O(N)).

### Space Complexity
- **O(N):**
  - The hash map stores a mapping of all nodes in the original list.

## Alternative Solution (Without Extra Space)
If you would like a solution with **O(1)** space complexity (besides the new nodes), the problem can be solved by interweaving the original and new nodes in a single list. Let me know if you'd like this approach to be explained in detail.
