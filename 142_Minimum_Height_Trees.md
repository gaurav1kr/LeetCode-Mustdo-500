# Minimum Height Trees - Optimized Solution

This document provides a concise and optimized C++ solution for the LeetCode problem [Minimum Height Trees](https://leetcode.com/problems/minimum-height-trees).

## Problem Overview
Given a tree with `n` nodes labeled from `0` to `n-1`, and an array of edges where `edges[i] = [a, b]` represents a bidirectional edge between nodes `a` and `b`, find all the roots of Minimum Height Trees (MHTs).

An MHT is defined as a rooted tree with the minimum height among all possible rooted trees formed from the given tree.

## Solution

### C++ Code
```cpp
#include <vector>
#include <queue>
using namespace std;

vector<int> findMinHeightTrees(int n, vector<vector<int>>& edges) {
    if (n == 1) return {0};
    vector<int> adj[n], degree(n, 0);
    for (const auto& edge : edges) {
        adj[edge[0]].push_back(edge[1]);
        adj[edge[1]].push_back(edge[0]);
        degree[edge[0]]++;
        degree[edge[1]]++;
    }
    
    queue<int> q;
    for (int i = 0; i < n; ++i)
        if (degree[i] == 1) q.push(i);

    while (n > 2) {
        int size = q.size();
        n -= size;
        for (int i = 0; i < size; ++i) {
            int node = q.front(); q.pop();
            for (int neighbor : adj[node]) {
                if (--degree[neighbor] == 1)
                    q.push(neighbor);
            }
        }
    }

    vector<int> result;
    while (!q.empty()) {
        result.push_back(q.front());
        q.pop();
    }
    return result;
}
```

### Explanation
1. **Special Case**:
   - If the tree has only one node (`n == 1`), return `[0]`.

2. **Graph Representation**:
   - Use an adjacency list to represent the tree.
   - Maintain an array `degree` to track the degree (number of connections) of each node.

3. **Topological Sorting**:
   - Use a queue to iteratively remove leaf nodes (nodes with degree 1).
   - For each removed leaf node, decrease the degree of its neighbors.
   - Add neighbors that become leaves to the queue.

4. **Termination**:
   - Stop when `n <= 2` nodes remain. These nodes are the roots of the MHTs.

5. **Result Construction**:
   - Collect the remaining nodes from the queue and return them as the result.

### Complexity
- **Time Complexity**: `O(n)`, where `n` is the number of nodes. Each edge and node is processed once.
- **Space Complexity**: `O(n)` for the adjacency list and the queue.

## Example
### Input
```text
n = 6
edges = [[3, 0], [3, 1], [3, 2], [3, 4], [5, 4]]
```
### Output
```text
[3, 4]
```

## Key Points
- The problem is solved using a topological sorting approach similar to Kahn's algorithm.
- This method efficiently finds the center(s) of the tree by iteratively peeling off leaves.

---

For more details, visit the [problem page](https://leetcode.com/problems/minimum-height-trees).
