
# Delete Node in a BST

## Problem
Given a root node reference of a BST and a key, delete the node with the given key in the BST. Return the root node reference of the BST (possibly updated).

You can find the problem details [here](https://leetcode.com/problems/delete-node-in-a-bst/description/).

---

## Optimized Solution (C++)

```cpp
class Solution {
public:
    TreeNode* deleteNode(TreeNode* root, int key) {
        if (!root) return nullptr; // Base case: Node not found
        if (key < root->val) {
            root->left = deleteNode(root->left, key); // Search in left subtree
        } else if (key > root->val) {
            root->right = deleteNode(root->right, key); // Search in right subtree
        } else {
            // Node found
            if (!root->left) return root->right; // No left child
            if (!root->right) return root->left; // No right child

            // Node with two children: Replace with inorder successor
            TreeNode* minNode = getMin(root->right);
            root->val = minNode->val;
            root->right = deleteNode(root->right, minNode->val); // Delete the successor
        }
        return root;
    }

private:
    TreeNode* getMin(TreeNode* node) {
        while (node->left) node = node->left;
        return node; // Find the leftmost (smallest) node
    }
};
```

---

## Explanation

1. **Base Case**:
   - If the `root` is `nullptr`, return `nullptr` (node not found).

2. **Traverse the Tree**:
   - If `key` is less than the current node's value, search the left subtree.
   - If `key` is greater, search the right subtree.

3. **Node Found**:
   - If the node has only one child, return the non-null child.
   - If the node has two children:
     - Find the inorder successor (smallest value in the right subtree).
     - Replace the current node's value with the inorder successor's value.
     - Delete the inorder successor.

4. **Helper Function**:
   - `getMin()` efficiently finds the smallest node in a subtree by traversing left.

---

## Complexity Analysis

- **Time Complexity**: `O(H)`
  - `H` is the height of the BST (worst-case `O(N)` for a skewed tree, best-case `O(log N)` for a balanced tree).

- **Space Complexity**: `O(H)`
  - Due to the recursive stack.
