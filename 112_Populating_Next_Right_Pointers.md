
# Populating Next Right Pointers in Each Node

### Problem Description

You are given a perfect binary tree where all leaves are on the same level, and every parent has two children. The objective is to connect each node's `next` pointer to its next right node. If there is no next right node, the `next` pointer should be set to `NULL`. The solution should use constant space (no additional data structures like queues or stacks).

---

### Optimized C++ Solution

```cpp
/*
// Definition for a Node.
class Node {
public:
    int val;
    Node* left;
    Node* right;
    Node* next;

    Node() : val(0), left(nullptr), right(nullptr), next(nullptr) {}
    Node(int _val) : val(_val), left(nullptr), right(nullptr), next(nullptr) {}
    Node(int _val, Node* _left, Node* _right, Node* _next)
        : val(_val), left(_left), right(_right), next(_next) {}
};
*/

class Solution {
public:
    Node* connect(Node* root) {
        if (!root) return nullptr;

        Node* currentLevel = root;

        // Traverse level by level
        while (currentLevel->left) {
            Node* current = currentLevel;

            // Connect nodes in the current level
            while (current) {
                current->left->next = current->right;
                if (current->next) {
                    current->right->next = current->next->left;
                }
                current = current->next;
            }

            // Move to the next level
            currentLevel = currentLevel->left;
        }

        return root;
    }
};
```

---

### Explanation

1. **Key Idea**:
   - Use the existing tree structure to connect the `next` pointers without additional memory.
   - Traverse level by level and connect children nodes.

2. **Algorithm**:
   - Start with the root node as `currentLevel`.
   - At each level, traverse the nodes using a `current` pointer.
   - Connect the `left` child of each node to its `right` child.
   - If the current node has a `next`, connect its `right` child to the `left` child of the next node.
   - Move `currentLevel` to the next level (the leftmost node of the current level).

3. **Perfect Binary Tree Property**:
   - The problem guarantees that the tree is perfect, which means all levels are fully populated, simplifying the traversal logic.

---

### Complexity Analysis

- **Time Complexity**: `O(n)`
  - Each node is visited exactly once.

- **Space Complexity**: `O(1)`
  - Only a few pointers are used for traversal, and no additional data structures are required.

---

### Example Usage

```cpp
#include <iostream>
using namespace std;

int main() {
    // Example tree setup can go here
    return 0;
}
```
