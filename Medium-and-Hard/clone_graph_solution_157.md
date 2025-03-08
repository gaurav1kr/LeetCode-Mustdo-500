# Clone Graph Solution

Here is an optimized and concise C++ solution for the LeetCode problem [Clone Graph](https://leetcode.com/problems/clone-graph/).

## Problem Description
Given a reference of a node in a **connected undirected graph**, return a **deep copy** (clone) of the graph.

Each node in the graph contains a `val` (int) and a list of its `neighbors` (List[Node]).

---

## Optimized C++ Solution

```cpp
#include <unordered_map>
#include <vector>

using namespace std;

// Definition for a Node.
class Node {
public:
    int val;
    vector<Node*> neighbors;
    Node() : val(0), neighbors(vector<Node*>()) {}
    Node(int _val) : val(_val), neighbors(vector<Node*>()) {}
    Node(int _val, vector<Node*> _neighbors) : val(_val), neighbors(_neighbors) {}
};

class Solution {
public:
    unordered_map<Node*, Node*> cloneMap;

    Node* cloneGraph(Node* node) {
        if (!node) return nullptr;

        // If node is already cloned, return the cloned node
        if (cloneMap.find(node) != cloneMap.end()) {
            return cloneMap[node];
        }

        // Create a clone of the current node
        Node* clone = new Node(node->val);
        cloneMap[node] = clone;

        // Recursively clone neighbors
        for (Node* neighbor : node->neighbors) {
            clone->neighbors.push_back(cloneGraph(neighbor));
        }

        return clone;
    }
};
```

---

## Explanation

### Key Points
1. **Hash Map Usage**:
   - `cloneMap` is used to store the mapping of original nodes to their cloned counterparts to handle cycles in the graph.

2. **DFS Recursion**:
   - The function recursively clones the graph by traversing neighbors.

3. **Base Case**:
   - Handles `nullptr` input for an empty graph.

---

## Complexity Analysis

- **Time Complexity**: \(O(V + E)\)
  - \(V\): Number of vertices (nodes).
  - \(E\): Number of edges.
  - Each node and edge is processed once.

- **Space Complexity**: \(O(V)\)
  - Due to the hash map (`cloneMap`) and recursion stack.

---

This implementation is efficient and handles edge cases such as:
- Cycles in the graph.
- Disconnected components.
- An empty graph (input as `nullptr`).
