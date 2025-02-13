
# Serialize and Deserialize Binary Tree

This markdown document provides an optimized C++ solution for the problem "Serialize and Deserialize Binary Tree" available on LeetCode.

## Problem Description

Design an algorithm to serialize and deserialize a binary tree. Serialization is the process of converting a tree to a single string, and deserialization is the process of converting the string back to the original tree structure.

The structure of the binary tree is defined as follows:

```cpp
struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};
```

## Optimized C++ Solution

```cpp
#include <string>
#include <sstream>
#include <queue>
using namespace std;

// Definition for a binary tree node.
struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

class Codec {
public:
    // Encodes a tree to a single string.
    string serialize(TreeNode* root) {
        if (!root) return "#";
        
        queue<TreeNode*> q;
        q.push(root);
        stringstream ss;
        
        while (!q.empty()) {
            TreeNode* node = q.front();
            q.pop();
            
            if (node) {
                ss << node->val << " ";
                q.push(node->left);
                q.push(node->right);
            } else {
                ss << "# ";
            }
        }
        
        return ss.str();
    }

    // Decodes your encoded data to tree.
    TreeNode* deserialize(string data) {
        if (data == "#") return nullptr;

        stringstream ss(data);
        string val;
        ss >> val;

        TreeNode* root = new TreeNode(stoi(val));
        queue<TreeNode*> q;
        q.push(root);

        while (!q.empty()) {
            TreeNode* node = q.front();
            q.pop();

            if (ss >> val && val != "#") {
                node->left = new TreeNode(stoi(val));
                q.push(node->left);
            }

            if (ss >> val && val != "#") {
                node->right = new TreeNode(stoi(val));
                q.push(node->right);
            }
        }

        return root;
    }
};

// Your Codec object will be instantiated and called as such:
// Codec codec;
// codec.deserialize(codec.serialize(root));
```

## Explanation

### Serialization
- Perform a level-order traversal using a queue.
- Append the value of each node to the output string.
- Use `"#"` to represent `null` nodes.
- Use spaces as delimiters to separate node values.

### Deserialization
- Use a queue to reconstruct the tree level by level.
- Start with the root node from the serialized string.
- For each node, read the next two values (left and right children) and add them to the queue if they are not `"#"`.

## Complexity Analysis

- **Time Complexity**: `O(n)` where `n` is the number of nodes in the tree. Both serialization and deserialization traverse the entire tree once.
- **Space Complexity**: `O(n)` for the queue used in level-order traversal during both serialization and deserialization.

## Example Usage

```cpp
// Example usage:
Codec codec;
TreeNode* root = new TreeNode(1);
root->left = new TreeNode(2);
root->right = new TreeNode(3);
root->right->left = new TreeNode(4);
root->right->right = new TreeNode(5);

string serialized = codec.serialize(root);
TreeNode* deserialized = codec.deserialize(serialized);
```
