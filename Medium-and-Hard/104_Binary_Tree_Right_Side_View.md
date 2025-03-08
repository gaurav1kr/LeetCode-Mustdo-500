
## Binary Tree Right Side View - Optimized Solution

Here is an optimized C++ solution for the [Binary Tree Right Side View](https://leetcode.com/problems/binary-tree-right-side-view/) problem on LeetCode. The solution uses a level-order traversal (BFS) approach, ensuring that the last node of each level is added to the result.

```cpp
#include <vector>
#include <queue>

using namespace std;

// Definition for a binary tree node.
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};

class Solution {
public:
    vector<int> rightSideView(TreeNode* root) {
        vector<int> result;
        if (!root) return result;

        queue<TreeNode*> q;
        q.push(root);

        while (!q.empty()) {
            int levelSize = q.size();
            for (int i = 0; i < levelSize; ++i) {
                TreeNode* node = q.front();
                q.pop();

                // If it's the last node in the current level, add it to the result
                if (i == levelSize - 1) {
                    result.push_back(node->val);
                }

                // Add left and right children to the queue
                if (node->left) q.push(node->left);
                if (node->right) q.push(node->right);
            }
        }

        return result;
    }
};
```

### Explanation

1. **Edge Case Handling**:
   - If the root is `nullptr`, the function immediately returns an empty result.

2. **BFS Approach**:
   - A queue is used to perform a level-order traversal.
   - For each level, the last node is added to the result.

3. **Time Complexity**:
   - O(N), where N is the number of nodes in the tree. Each node is processed once.

4. **Space Complexity**:
   - O(W), where W is the maximum width of the tree (at most O(N) in the worst case for a completely balanced tree).

This solution is efficient and follows the principles of clean, maintainable code.
