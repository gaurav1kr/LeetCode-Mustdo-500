# Path Sum III - Optimal C++ Solution

This markdown file explains the optimal C++ solution for the **Path Sum III** problem on LeetCode. The approach uses the **prefix sum technique** combined with a hashmap to achieve an efficient solution.

## Problem Description
You are given the root of a binary tree and an integer `targetSum`. Return the number of paths where the sum of the values along the path equals `targetSum`.

The path does not need to start or end at the root or a leaf, but it must go downwards (i.e., traveling only from parent nodes to child nodes).

### Constraints:
- The number of nodes in the tree is in the range `[0, 10^4]`.
- `-10^9 <= Node.val <= 10^9`
- `-1000 <= targetSum <= 1000`

## Optimal Solution

### Code:
```cpp
#include <unordered_map>
using namespace std;

struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

class Solution {
public:
    int pathSum(TreeNode* root, int targetSum) {
        unordered_map<long long, int> prefixSumCount; // Use long long for prefix sums
        prefixSumCount[0] = 1; // Base case: one path with sum 0 (empty path)
        return dfs(root, 0, targetSum, prefixSumCount);
    }

private:
    int dfs(TreeNode* node, long long currentSum, int targetSum, unordered_map<long long, int>& prefixSumCount) {
        if (!node) return 0;

        currentSum += node->val; // Add the current node's value to the running sum
        // Check how many times (currentSum - targetSum) has occurred
        int paths = prefixSumCount[currentSum - targetSum];

        // Update the prefix sum count for the current sum
        prefixSumCount[currentSum]++;

        // Recur for left and right subtrees
        paths += dfs(node->left, currentSum, targetSum, prefixSumCount);
        paths += dfs(node->right, currentSum, targetSum, prefixSumCount);

        // Backtrack: remove the current sum from the map after exploring the node
        prefixSumCount[currentSum]--;

        return paths;
    }
};
```

## Explanation
### Prefix Sum Concept:
The prefix sum is the cumulative sum of all node values from the root to the current node. To find if there exists a path with a sum equal to `targetSum`, we calculate:

```
prefixSum[currentSum] = prefixSum[currentSum - targetSum]
```

### Steps:
1. Traverse the tree using **DFS**.
2. Maintain a running sum of node values (`currentSum`).
3. Use a hashmap (`prefixSumCount`) to store the frequency of each prefix sum.
4. For each node, calculate how many times `currentSum - targetSum` exists in the hashmap. This gives the count of paths ending at the current node that sum to `targetSum`.
5. Recur for left and right child nodes.
6. Use backtracking to remove the current node's contribution from the hashmap before returning to the parent.

### Key Points:
- **Base Case**: The empty path (prefix sum = 0) is initialized with a frequency of 1.
- **Backtracking** ensures that the hashmap does not retain outdated values from sibling paths.

## Complexity
- **Time Complexity**: `O(N)` where `N` is the number of nodes in the tree. Each node is visited once.
- **Space Complexity**: `O(H)`, where `H` is the height of the tree (for the recursion stack and hashmap).

## Fix for Integer Overflow
The problem constraints include large values for node values and running sums, so integer overflow can occur. To handle this:
- Use `long long` for `currentSum` and the keys in `prefixSumCount`.

## Edge Cases
1. The tree is empty (`root == nullptr`): Return `0`.
2. Large positive or negative values for node values.
3. The path includes a single node that matches `targetSum`.
4. Multiple overlapping paths summing to `targetSum`.

## Example Usage
### Input:
```
Tree:       10
           /  \
          5   -3
         / \     \
        3   2    11
       / \   \
      3  -2   1
TargetSum: 8
```

### Output:
```
3
```
Explanation: The paths are:
- 5 -> 3
- 5 -> 2 -> 1
- -3 -> 11
