## Validate Binary Search Tree (LeetCode)

### **Optimal C++ Solution**

This solution uses a recursive approach with a range check to ensure the given binary tree is a valid Binary Search Tree (BST).

### **C++ Code:**
```cpp
#include <limits.h>
#include <iostream>

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

class Solution {
public:
    bool isValidBST(TreeNode* root, long minVal = LONG_MIN, long maxVal = LONG_MAX) {
        if (!root) return true; // An empty tree is a valid BST
        if (root->val <= minVal || root->val >= maxVal) return false;
        
        return isValidBST(root->left, minVal, root->val) &&
               isValidBST(root->right, root->val, maxVal);
    }
};
```

### **Explanation:**
1. **Base Case:** If the tree is empty (`nullptr`), it is a valid BST.
2. **Validity Check:** Each node's value must lie within a valid range `(minVal, maxVal)`.
3. **Recursive Calls:**
   - The left subtree must have values in the range `(minVal, root->val)`.
   - The right subtree must have values in the range `(root->val, maxVal)`.
4. **Time Complexity:** `O(n)`, where `n` is the number of nodes (each node is visited once).
5. **Space Complexity:** `O(h)`, where `h` is the height of the tree (due to recursive stack calls). In a balanced tree, this is `O(log n)`, while in a skewed tree, it’s `O(n)`.

### **Key Takeaways:**
- Uses a range-based validation approach.
- Efficient with `O(n)` time complexity.
- Recursive function ensures each subtree satisfies BST properties.

This approach effectively validates a BST in an optimal manner. 🚀
