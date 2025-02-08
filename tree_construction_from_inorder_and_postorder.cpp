# Construct Binary Tree from Inorder and Postorder Traversal

## Problem Description
Given two integer arrays `inorder` and `postorder` where:
- `inorder` is the inorder traversal of a binary tree.
- `postorder` is the postorder traversal of the same tree.

Construct and return the binary tree.

---

## Optimized C++ Solution

### Explanation:
1. **Inorder Traversal**: Left -> Root -> Right
2. **Postorder Traversal**: Left -> Right -> Root

The last element of the `postorder` array is always the root of the tree. Using a map for `inorder` traversal allows efficient lookup for the root's index.

---

### Code:
```cpp
#include <iostream>
#include <vector>
#include <unordered_map>

using namespace std;

// Definition for a binary tree node.
struct TreeNode 
{
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

class Solution 
{
public:
    TreeNode* buildTree(vector<int>& inorder, vector<int>& postorder) 
    {
        // Map to store the index of each value in inorder traversal
        unordered_map<int, int> inorderIndexMap;
        for (int i = 0; i < inorder.size(); i++) 
	{
            inorderIndexMap[inorder[i]] = i;
        }

        // Start building the tree
        int postIndex = postorder.size() - 1;
        return constructTree(inorder, postorder, inorderIndexMap, postIndex, 0, inorder.size() - 1);
    }

private:
    TreeNode* constructTree(const vector<int>& inorder, const vector<int>& postorder,
                            unordered_map<int, int>& inorderIndexMap, int& postIndex,
                            int inStart, int inEnd) {
        // Base case: no elements to construct the subtree
        if (inStart > inEnd) return nullptr;

        // Get the current root value and create a TreeNode
        int rootValue = postorder[postIndex--];
        TreeNode* root = new TreeNode(rootValue);

        // Find the index of the root value in the inorder array
        int rootIndex = inorderIndexMap[rootValue];

        // Recursively build the right and left subtrees
        root->right = constructTree(inorder, postorder, inorderIndexMap, postIndex, rootIndex + 1, inEnd);
        root->left = constructTree(inorder, postorder, inorderIndexMap, postIndex, inStart, rootIndex - 1);

        return root;
    }
};
```

---

### Complexity:
- **Time Complexity**: `O(n)`, where `n` is the number of nodes in the tree. The use of a hashmap allows us to find the index of the root in constant time.
- **Space Complexity**: `O(n)` for the hashmap and recursion stack in the worst case (e.g., a skewed tree).

---
