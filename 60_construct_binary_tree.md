# Construct Binary Tree from Preorder and Inorder Traversal

## Problem Description
Given two integer arrays `preorder` and `inorder` where:
- `preorder` is the preorder traversal of a binary tree.
- `inorder` is the inorder traversal of the same tree.

Construct and return the binary tree.

## Approach
1. The **preorder traversal** gives the root node first, followed by left and right subtrees.
2. The **inorder traversal** helps determine the left and right subtrees by locating the root node.
3. Use a **hash map** to store the indices of inorder values for quick lookup.
4. Recursively build the left and right subtrees.

## C++ Solution

```cpp
#include <unordered_map>
#include <vector>

using namespace std;

// Definition for a binary tree node.
struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

class Solution {
public:
    unordered_map<int, int> inorderMap;
    int preorderIndex = 0;

    TreeNode* buildTreeHelper(vector<int>& preorder, vector<int>& inorder, int left, int right) {
        if (left > right) return nullptr;

        // Get the root value from preorder traversal
        int rootVal = preorder[preorderIndex++];
        TreeNode* root = new TreeNode(rootVal);

        // Find root index in inorder traversal
        int inorderIndex = inorderMap[rootVal];

        // Build left and right subtrees
        root->left = buildTreeHelper(preorder, inorder, left, inorderIndex - 1);
        root->right = buildTreeHelper(preorder, inorder, inorderIndex + 1, right);

        return root;
    }

    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        // Store inorder values with their indices for quick lookup
        for (int i = 0; i < inorder.size(); i++) {
            inorderMap[inorder[i]] = i;
        }
        return buildTreeHelper(preorder, inorder, 0, inorder.size() - 1);
    }
};
```

## Complexity Analysis
- **Time Complexity**: \(O(n)\) (Each node is processed once, and lookup in `unordered_map` is \(O(1)\)).
- **Space Complexity**: \(O(n)\) (For storing the hash map and recursive call stack in the worst case).

## Example Usage

```cpp
int main() {
    Solution solution;
    vector<int> preorder = {3, 9, 20, 15, 7};
    vector<int> inorder = {9, 3, 15, 20, 7};

    TreeNode* root = solution.buildTree(preorder, inorder);

    return 0;
}
```

This solution efficiently constructs a binary tree from given traversal sequences. 🚀
