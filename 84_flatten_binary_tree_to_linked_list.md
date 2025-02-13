
# Flatten Binary Tree to Linked List

## Problem Description
Given the root of a binary tree, flatten the tree into a "linked list":
- The "linked list" should use the same TreeNode class where the `right` child pointer points to the next node in the list and the `left` child pointer is always `nullptr`.
- The "linked list" should be in the same order as a pre-order traversal of the binary tree.

## Optimal Solution (Recursive)

Here’s an optimal C++ solution that modifies the tree in-place without using extra space:

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    void flatten(TreeNode* root) {
        TreeNode* prev = nullptr;

        function<void(TreeNode*)> dfs = [&](TreeNode* node) {
            if (!node) return;

            // Process the right subtree first
            dfs(node->right);
            // Then process the left subtree
            dfs(node->left);

            // Flatten the current node
            node->right = prev;
            node->left = nullptr;
            prev = node;
        };

        dfs(root);
    }
};
```

### Explanation
1. **Approach**:
   - Traverse the binary tree in reverse post-order (right → left → root).
   - Use a `prev` pointer to keep track of the previously processed node.
   - Set the `right` pointer of the current node to `prev` and the `left` pointer to `nullptr`.
   - Update `prev` to the current node.

2. **Advantages**:
   - **In-place**: The solution doesn’t require extra space as it modifies the tree directly.
   - **Efficient**: Each node is visited exactly once, so the time complexity is \(O(n)\), where \(n\) is the number of nodes in the tree.

---

## Alternative Solution (Iterative)

Here’s an alternative approach using an iterative method:

```cpp
class Solution {
public:
    void flatten(TreeNode* root) {
        TreeNode* curr = root;

        while (curr) {
            if (curr->left) {
                // Find the rightmost node of the left subtree
                TreeNode* rightmost = curr->left;
                while (rightmost->right) {
                    rightmost = rightmost->right;
                }

                // Link right subtree to the rightmost node of the left subtree
                rightmost->right = curr->right;

                // Move left subtree to the right
                curr->right = curr->left;
                curr->left = nullptr;
            }
            // Move to the next node
            curr = curr->right;
        }
    }
};
```

### Comparison of Approaches
| Approach    | Space Complexity | Time Complexity | Notes                               |
|-------------|------------------|-----------------|-------------------------------------|
| Recursive   | \(O(1)\)       | \(O(n)\)      | Uses reverse post-order traversal. |
| Iterative   | \(O(1)\)       | \(O(n)\)      | Handles stack limitations.          |

---

Both solutions are efficient and modify the tree in-place without extra space. Choose the recursive one for cleaner code or the iterative one for stack-constrained environments.
