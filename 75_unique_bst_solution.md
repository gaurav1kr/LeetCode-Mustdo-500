
# Optimal C++ Solution for LeetCode Problem: Unique Binary Search Trees

## Problem Description

The problem [Unique Binary Search Trees](https://leetcode.com/problems/unique-binary-search-trees/) asks us to determine the number of unique binary search trees (BSTs) that can be formed with `n` distinct nodes labeled from `1` to `n`.

The problem can be solved using dynamic programming with the following recurrence relation:

\[ G(n) = \sum_{i=1}^{n} G(i-1) \cdot G(n-i) \]

Where:
- \( G(0) = 1 \): An empty tree.
- \( G(1) = 1 \): A tree with one node.

---

## Solution

Below is the optimal C++ implementation of the solution using dynamic programming:

```cpp
#include <vector>

class Solution {
public:
    int numTrees(int n) {
        std::vector<int> dp(n + 1, 0);
        dp[0] = 1; // Base case: Empty tree
        dp[1] = 1; // Base case: Tree with one node

        for (int i = 2; i <= n; ++i) { // Compute dp[i] for all 2 <= i <= n
            for (int j = 1; j <= i; ++j) { // Consider each node as root
                dp[i] += dp[j - 1] * dp[i - j];
            }
        }

        return dp[n];
    }
};
```

---

## Complexity Analysis

### Time Complexity
- **\( O(n^2) \)**
  - For each \( i \), we iterate through \( j = 1 \) to \( i \), resulting in \( O(n^2) \) operations.

### Space Complexity
- **\( O(n) \)**
  - For storing the DP array of size \( n+1 \).

---

## Example Usage

```cpp
#include <iostream>
using namespace std;

int main() {
    Solution solution;
    int n = 3; // Input
    cout << "Number of unique BSTs for n = " << n << " is: " << solution.numTrees(n) << endl;
    return 0;
}
```

### Output for `n = 3`:
```
Number of unique BSTs for n = 3 is: 5
```

---

## Explanation of Approach

1. Use a dynamic programming array `dp` where `dp[i]` stores the number of unique BSTs that can be formed with `i` nodes.
2. Initialize `dp[0] = 1` and `dp[1] = 1` as base cases.
3. For each `i` from 2 to `n`, compute the value of `dp[i]` using the recurrence relation:
   - Choose each node `j` (from 1 to `i`) as the root.
   - Multiply the number of unique BSTs possible with `j-1` nodes (left subtree) and `i-j` nodes (right subtree).
4. Return `dp[n]` as the result.

---

This approach efficiently calculates the number of unique BSTs using dynamic programming. Let me know if further clarifications are needed!
