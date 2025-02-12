# C++ Optimized Solution for Kth Smallest Element in a BST

Here is an optimized solution for finding the kth smallest element in a Binary Search Tree (BST) using C++. The approach uses an in-order traversal to efficiently find the required element.

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
    int kthSmallest(TreeNode* root, int k) {
        int count = 0;
        int result = 0;
        inOrderTraversal(root, k, count, result);
        return result;
    }

private:
    void inOrderTraversal(TreeNode* node, int k, int& count, int& result) {
        if (!node) return;

        // Traverse the left subtree
        inOrderTraversal(node->left, k, count, result);

        // Increment the count and check if it's the kth element
        count++;
        if (count == k) {
            result = node->val;
            return;
        }

        // Traverse the right subtree
        inOrderTraversal(node->right, k, count, result);
    }
};
```

## Explanation

### Approach
1. **In-Order Traversal**: Perform an in-order traversal of the BST. This visits the nodes in ascending order.
2. **Stop Early**: Stop the traversal as soon as the kth smallest element is found, avoiding unnecessary computation.

### Complexity Analysis
- **Time Complexity**: O(H + k), where H is the height of the tree. The traversal only processes k nodes and the height of the tree for recursion.
- **Space Complexity**: O(H), due to the recursion stack for the in-order traversal.

### Steps in the Code
1. Use a helper function `inOrderTraversal` to perform the recursive in-order traversal.
2. Maintain a `count` variable to keep track of the number of nodes visited so far.
3. As soon as `count == k`, update the result and terminate further traversal.

This implementation ensures an efficient and clean solution to the problem.
