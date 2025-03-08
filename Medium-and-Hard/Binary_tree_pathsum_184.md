
# Path Sum II - Optimized C++ Solution

Here is an optimized C++ solution for the "Path Sum II" problem on LeetCode, using Depth-First Search (DFS). The solution avoids unnecessary vector copies during recursion to improve performance.

```cpp
#include <vector>
#include <iostream>
#include <algorithm>
using namespace std;

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
    void findPaths(TreeNode* node, int targetSum, vector<int>& currentPath, vector<vector<int>>& result) {
        if (!node) return;

        // Add current node's value to the path
        currentPath.push_back(node->val);
        
        // Check if it's a leaf node and the path sum equals the target sum
        if (!node->left && !node->right && targetSum == node->val) {
            result.push_back(currentPath);
        } else {
            // Recurse to left and right children
            findPaths(node->left, targetSum - node->val, currentPath, result);
            findPaths(node->right, targetSum - node->val, currentPath, result);
        }

        // Backtrack by removing the current node's value from the path
        currentPath.pop_back();
    }

    vector<vector<int>> pathSum(TreeNode* root, int targetSum) {
        vector<vector<int>> result;
        vector<int> currentPath;
        findPaths(root, targetSum, currentPath, result);
        return result;
    }
};

// Helper function to create a tree for testing purposes
TreeNode* createTree(vector<int> values, int index) {
    if (index >= values.size() || values[index] == -1) return nullptr;
    TreeNode* root = new TreeNode(values[index]);
    root->left = createTree(values, 2 * index + 1);
    root->right = createTree(values, 2 * index + 2);
    return root;
}

// Example usage
int main() {
    vector<int> treeValues = {5, 4, 8, 11, -1, 13, 4, 7, 2, -1, -1, 5, 1};
    int targetSum = 22;
    TreeNode* root = createTree(treeValues, 0);

    Solution solution;
    vector<vector<int>> result = solution.pathSum(root, targetSum);

    for (const auto& path : result) {
        for (int val : path) {
            cout << val << " ";
        }
        cout << endl;
    }

    return 0;
}
```

### Explanation:

1. **`findPaths` Function**:
   - It performs a DFS to explore all possible root-to-leaf paths.
   - It adds the current node's value to the path and recursively explores left and right children.
   - If a leaf node is reached and the target sum matches, the path is added to the result.
   - Backtracking is done by removing the last node from `currentPath`.

2. **Avoiding Vector Copy**:
   - By passing `currentPath` as a reference and backtracking, unnecessary copies of vectors are avoided, improving performance.

3. **Tree Construction**:
   - The `createTree` function helps construct a binary tree for testing using level-order representation.

This code is optimized and works efficiently for typical constraints in the problem.
