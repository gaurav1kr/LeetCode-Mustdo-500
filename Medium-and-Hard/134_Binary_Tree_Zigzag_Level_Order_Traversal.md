
# Binary Tree Zigzag Level Order Traversal - C++ Solution

Here is a concise and optimized C++ solution for the **Binary Tree Zigzag Level Order Traversal** problem on LeetCode:

```cpp
#include <vector>
#include <queue>
#include <deque>
using namespace std;

struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

vector<vector<int>> zigzagLevelOrder(TreeNode* root) {
    if (!root) return {};
    
    vector<vector<int>> result;
    queue<TreeNode*> q;
    q.push(root);
    bool leftToRight = true;

    while (!q.empty()) {
        int size = q.size();
        deque<int> level;

        for (int i = 0; i < size; ++i) {
            TreeNode* node = q.front();
            q.pop();

            if (leftToRight)
                level.push_back(node->val);
            else
                level.push_front(node->val);

            if (node->left) q.push(node->left);
            if (node->right) q.push(node->right);
        }

        result.emplace_back(level.begin(), level.end());
        leftToRight = !leftToRight;
    }

    return result;
}
```

## Key Features:
1. **Breadth-First Search**: The solution uses a queue for level-order traversal.
2. **Deque for Zigzag**: A `deque` is used to efficiently insert elements at both ends depending on the current traversal direction.
3. **Toggle Direction**: The `leftToRight` boolean flag toggles after processing each level.

### Complexity:
- **Time Complexity**: \(O(N)\), where \(N\) is the number of nodes in the tree.
- **Space Complexity**: \(O(N)\), due to the storage of nodes in the queue and deque.
