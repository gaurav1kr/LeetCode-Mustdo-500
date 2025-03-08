# House Robber III - Optimized C++ Solution

This markdown file contains the optimized solution for the "House Robber III" problem from LeetCode, along with an explanation and complexity analysis.

## Problem Description

Given a binary tree, each node in the tree contains a value representing the amount of money stored in it. You are tasked to rob the tree, such that no two directly-connected nodes are robbed. Your goal is to return the maximum amount of money you can rob.

---

## Optimized C++ Code

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
    // Helper function to return a pair of values
    // First value: Maximum sum if the current node is NOT robbed
    // Second value: Maximum sum if the current node IS robbed
    pair<int, int> robSub(TreeNode* root) {
        if (!root) return {0, 0};
        
        // Recursively calculate for left and right subtrees
        auto left = robSub(root->left);
        auto right = robSub(root->right);
        
        // If we do not rob the current node, we can take the max of both states (robbed or not) from its children
        int notRobbed = max(left.first, left.second) + max(right.first, right.second);
        
        // If we rob the current node, we cannot rob its immediate children
        int robbed = root->val + left.first + right.first;
        
        return {notRobbed, robbed};
    }
    
    int rob(TreeNode* root) {
        auto result = robSub(root);
        return max(result.first, result.second);
    }
};
```

---

## Explanation

1. **Tree Traversal**:  
   - The function `robSub` performs a **post-order traversal** (processes left and right subtrees before processing the current node).

2. **Dynamic Programming**:  
   - For each node, compute two values:
     - **notRobbed**: Maximum sum if the current node is not robbed. This allows robbing its children.
     - **robbed**: Maximum sum if the current node is robbed. This prevents robbing its children.
   - Return these two values as a pair.

3. **Base Case**:  
   - If the node is `nullptr`, return `{0, 0}` because there is nothing to rob.

4. **Final Result**:  
   - For the root node, the maximum amount that can be robbed is the maximum of `notRobbed` and `robbed`.

---

## Complexity Analysis

- **Time Complexity**: \(O(n)\), where \(n\) is the number of nodes in the tree. Each node is visited once.
- **Space Complexity**: \(O(h)\), where \(h\) is the height of the tree. This is the stack space used during recursion.

---

## Notes

- This solution uses the principle of Dynamic Programming combined with post-order traversal to solve the problem efficiently.
- The use of a `pair<int, int>` helps in keeping track of both states (robbed or not) for each node, reducing the need for additional data structures.
