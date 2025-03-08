
# Binary Tree Level Order Traversal

## Problem Description
Given the root of a binary tree, return the level order traversal of its nodes' values. (i.e., from left to right, level by level).

**LeetCode Link**: [Binary Tree Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/description/)

---

## C++ Solution
```cpp
#include <iostream>
#include <vector>
#include <queue>
using namespace std;

// Definition for a binary tree node.
struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode* left, TreeNode* right) : val(x), left(left), right(right) {}
};

class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        vector<vector<int>> result;
        if (!root) return result;

        queue<TreeNode*> q;
        q.push(root);

        while (!q.empty()) {
            int size = q.size();
            vector<int> level;

            for (int i = 0; i < size; ++i) {
                TreeNode* node = q.front();
                q.pop();
                level.push_back(node->val);

                if (node->left) q.push(node->left);
                if (node->right) q.push(node->right);
            }

            result.push_back(level);
        }

        return result;
    }
};
```

---

## Explanation
1. **Initialization**:
   - Check if the root is null; if so, return an empty result.
   - Use a queue to perform BFS. Push the root node into the queue.

2. **BFS Traversal**:
   - Traverse each level of the tree while the queue is not empty.
   - For each level:
     - Store the size of the queue (`size`) to iterate through all nodes in the current level.
     - Process `size` nodes by popping from the queue, adding their values to the current level vector, and pushing their children (if any) into the queue.

3. **Result Construction**:
   - After processing all nodes at a level, add the level vector to the result.

4. **Return the Result**:
   - After traversing all levels, return the result vector.

---

## Complexity
- **Time Complexity**: \(O(n)\), where \(n)\) is the number of nodes in the tree. Each node is visited once.
- **Space Complexity**: \(O(n)\), due to the queue storage for the nodes at the deepest level of the tree.

---

## Example
### Input:
```
    3
   / \
  9  20
     /  \
    15   7
```

### Output:
```
[
  [3],
  [9,20],
  [15,7]
]
```
