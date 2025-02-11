
# LeetCode Problem: Unique Binary Search Trees II

## Problem Description

Given an integer `n`, return all the structurally unique BST's (binary search trees), which have exactly `n` nodes of unique values from 1 to `n`. Return the answer in any order.

### Example:

#### Input:
```
n = 3
```
#### Output:
```
[[1,null,2,null,3],[1,null,3,2],[2,1,3],[3,1,null,null,2],[3,2,null,1]]
```

---

## Optimized C++ Solution

Here is a concise and optimized C++ solution for the problem:

```cpp
#include <vector>
using namespace std;

// Definition for a binary tree node.
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

class Solution {
public:
    vector<TreeNode*> generateTrees(int n) {
        if (n == 0) return {};
        return buildTrees(1, n);
    }
    
private:
    vector<TreeNode*> buildTrees(int start, int end) {
        if (start > end) return {nullptr};
        
        vector<TreeNode*> result;
        for (int i = start; i <= end; ++i) {
            vector<TreeNode*> leftTrees = buildTrees(start, i - 1);
            vector<TreeNode*> rightTrees = buildTrees(i + 1, end);
            
            for (auto left : leftTrees) {
                for (auto right : rightTrees) {
                    TreeNode* root = new TreeNode(i);
                    root->left = left;
                    root->right = right;
                    result.push_back(root);
                }
            }
        }
        return result;
    }
};
```

---

## Explanation

1. **Recursive Function `buildTrees`**:
   - This function generates all unique BSTs for a given range `[start, end]`.
   - Base case: If `start > end`, return a vector with `nullptr`.

2. **Divide and Conquer**:
   - For each root value `i` in the range `[start, end]`:
     - Recursively generate all possible left subtrees (`[start, i - 1]`).
     - Recursively generate all possible right subtrees (`[i + 1, end]`).
   - Combine each left subtree with each right subtree to form unique trees with `i` as the root.

3. **Final Output**:
   - Call `buildTrees(1, n)` to generate all trees for the range `[1, n]`.

---

## Complexity Analysis

- **Time Complexity**: 
  - The time complexity is `O(C_n)` where `C_n` is the nth Catalan number, as it determines the number of unique BSTs for `n` nodes.
  - Catalan numbers grow exponentially, so this is asymptotically exponential.

- **Space Complexity**: 
  - `O(C_n)` due to the recursive call stack and storing tree nodes.

---

## Notes

- The code uses recursion effectively to build trees for smaller ranges and combines them.
- Each tree node is dynamically allocated using `new`, so memory management (if necessary) must be handled carefully in a real-world implementation.
- This solution is clear, concise, and adheres to the problem constraints.
