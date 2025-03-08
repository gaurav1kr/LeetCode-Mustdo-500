
# Solution for LeetCode Problem: All Nodes Distance K in Binary Tree

## Problem Description
You are given the root of a binary tree, a target node, and an integer `k`. Your task is to find all nodes that are at distance `k` from the target node.

---

## Optimized C++ Solution
Below is the concise and optimized solution in C++:

```cpp
#include <vector>
#include <unordered_map>
#include <unordered_set>

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
    unordered_map<TreeNode*, TreeNode*> parentMap;

    void buildParentMap(TreeNode* root, TreeNode* parent) {
        if (!root) return;
        parentMap[root] = parent;
        buildParentMap(root->left, root);
        buildParentMap(root->right, root);
    }
    
public:
    vector<int> distanceK(TreeNode* root, TreeNode* target, int k) {
        buildParentMap(root, nullptr);
        unordered_set<TreeNode*> visited;
        vector<int> result;
        queue<pair<TreeNode*, int>> q;
        q.push({target, 0});
        visited.insert(target);

        while (!q.empty()) {
            auto [node, dist] = q.front();
            q.pop();

            if (dist == k) {
                result.push_back(node->val);
            } else if (dist < k) {
                for (TreeNode* neighbor : {node->left, node->right, parentMap[node]}) {
                    if (neighbor && !visited.count(neighbor)) {
                        visited.insert(neighbor);
                        q.push({neighbor, dist + 1});
                    }
                }
            }
        }
        return result;
    }
};
```

---

## Key Features of the Solution
1. **Parent Map Construction**:
   - A `buildParentMap` function uses DFS to track parent pointers for each node in the tree.
   - This allows traversal both upwards (to the parent) and downwards (to the children).

2. **Breadth-First Search (BFS)**:
   - BFS starts from the target node to explore all nodes at increasing distances.
   - Stops processing nodes when the required distance `k` is reached.

3. **Efficient Node Tracking**:
   - An `unordered_set` tracks visited nodes to avoid revisiting them.

---

## Complexity Analysis
- **Time Complexity**: `O(N)`
  - The DFS for building the parent map traverses all nodes once.
  - The BFS for finding nodes at distance `k` also traverses each node at most once.

- **Space Complexity**: `O(N)`
  - Space is used for the parent map, visited set, and BFS queue.

---

## Example Usage

```cpp
int main() {
    TreeNode* root = new TreeNode(3);
    root->left = new TreeNode(5);
    root->right = new TreeNode(1);
    root->left->left = new TreeNode(6);
    root->left->right = new TreeNode(2);
    root->right->left = new TreeNode(0);
    root->right->right = new TreeNode(8);

    TreeNode* target = root->left; // Target is node with value 5
    int k = 2;

    Solution sol;
    vector<int> result = sol.distanceK(root, target, k);

    // Output the result
    for (int val : result) {
        cout << val << " ";
    }
    return 0;
}
```

---

## Notes
- This solution is well-suited for binary trees of varying shapes and sizes.
- Handles edge cases such as when the tree is empty, `k` is 0, or `k` is greater than the tree height.

For more details, visit the [LeetCode problem page](https://leetcode.com/problems/all-nodes-distance-k-in-binary-tree).
