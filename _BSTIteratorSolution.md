# Binary Search Tree Iterator

## Problem Description
The goal is to implement an iterator for a binary search tree (BST) that supports the following operations efficiently:

1. `next()` - Returns the next smallest number in the BST.
2. `hasNext()` - Returns whether there is a next smallest number.

The solution should optimize for average \(O(1)\) time complexity for `next()` and \(O(1)\) for `hasNext()`.

## Optimized C++ Solution
The following implementation uses a stack to perform an in-order traversal of the BST efficiently:

```cpp
class BSTIterator {
private:
    stack<TreeNode*> st; // Stack to hold nodes during traversal

    // Helper function to push all left nodes onto the stack
    void pushLeft(TreeNode* node) {
        while (node) {
            st.push(node);
            node = node->left;
        }
    }

public:
    // Constructor initializes the stack with the leftmost path
    BSTIterator(TreeNode* root) {
        pushLeft(root);
    }

    // Return the next smallest number in the BST
    int next() {
        TreeNode* topNode = st.top();
        st.pop();
        if (topNode->right) {
            pushLeft(topNode->right);
        }
        return topNode->val;
    }

    // Return whether there are more nodes to traverse
    bool hasNext() {
        return !st.empty();
    }
};
```

## Explanation
1. **Initialization**: 
   - In the constructor, all the left children of the root are pushed onto the stack. This ensures the smallest element is on top of the stack.

2. **`next()`**:
   - Pop the top node from the stack (smallest element in the current traversal).
   - If the popped node has a right child, push all its left descendants onto the stack.

3. **`hasNext()`**:
   - Simply checks if the stack is non-empty, meaning there are more nodes to traverse.

## Complexity Analysis
- **Time Complexity**:
  - `next()`: Average \(O(1)\). Each node is pushed and popped from the stack exactly once.
  - `hasNext()`: \(O(1)\), as it only checks the stack's emptiness.
- **Space Complexity**:
  - \(O(h)\), where \(h\) is the height of the tree. The stack stores at most \(h\) nodes.

This implementation is efficient and adheres to the problem's constraints.
