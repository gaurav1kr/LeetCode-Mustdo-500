## Binary Tree Maximum Path Sum Solution

### Problem Statement
Given a **binary tree**, find the **maximum path sum**. A **path** in a binary tree is a sequence of nodes where each pair of adjacent nodes in the sequence has an edge connecting them. A node can only appear **once** in the path, but it does not need to go through the root. The path sum is the **sum of all node values** in the path.

### Approach
The solution employs a **Depth-First Search (DFS)** technique with recursion to traverse the tree and compute the maximum path sum.

### Algorithm
1. Define a helper function `maxGain(TreeNode* node, int& maxSum)` that:
   - Recursively computes the **maximum gain** from the left and right subtrees.
   - Ignores negative path sums to maximize contributions.
   - Calculates the **new potential path sum** by including the current node.
   - Updates `maxSum` with the maximum encountered value.
   - Returns the **maximum path sum** that includes the current node and either its left or right subtree.
2. Initialize `maxSum` with **INT_MIN**.
3. Call `maxGain` on the root node to start the computation.
4. Return `maxSum` as the final result.

### Code Implementation
```cpp
#include <bits/stdc++.h>
using namespace std;

// Definition for a binary tree node.
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

class Solution {
public:
    int maxPathSum(TreeNode* root) {
        int maxSum = INT_MIN;
        maxGain(root, maxSum);
        return maxSum;
    }
    
private:
    int maxGain(TreeNode* node, int& maxSum) {
        if (!node) return 0;
        
        // Recursively calculate max sum from left and right subtrees
        int leftGain = max(0, maxGain(node->left, maxSum)); // Ignore negative paths
        int rightGain = max(0, maxGain(node->right, maxSum));
        
        // Compute max path sum considering current node as the root
        int newPathSum = node->val + leftGain + rightGain;
        
        // Update global maxSum if new path sum is greater
        maxSum = max(maxSum, newPathSum);
        
        // Return max gain including the current node
        return node->val + max(leftGain, rightGain);
    }
};
```

### Complexity Analysis
- **Time Complexity:** **O(N)**, where N is the number of nodes in the tree. Each node is visited once.
- **Space Complexity:** **O(H)**, where H is the height of the tree (due to recursion stack in worst case).

### Edge Cases Considered
- A tree with a single node.
- Trees where all node values are negative.
- Unbalanced trees with varying subtree sizes.

### Summary
- Uses **DFS** with a helper function to compute max path sums efficiently.
- Avoids negative contributions to maximize the result.
- Returns the maximum sum found in the tree path.

This approach ensures an **optimal solution** with **O(N) complexity** while maintaining clarity and efficiency. 🚀
