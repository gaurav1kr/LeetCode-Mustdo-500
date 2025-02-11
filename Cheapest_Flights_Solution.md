
# LeetCode Problem: Cheapest Flights Within K Stops

## Problem Description
The problem can be found at [LeetCode - Cheapest Flights Within K Stops](https://leetcode.com/problems/cheapest-flights-within-k-stops/).

You are tasked to find the cheapest price from a starting city `src` to a destination city `dst` with at most `k` stops. If no route satisfies the condition, return `-1`.

## Optimized Solution

This solution uses a **Priority Queue (Dijkstra-like Algorithm)** approach to efficiently compute the result.

### C++ Implementation
```cpp
#include <vector>
#include <queue>
#include <climits>
using namespace std;

int findCheapestPrice(int n, vector<vector<int>>& flights, int src, int dst, int k) {
    // Adjacency list to store graph: {destination, cost}
    vector<vector<pair<int, int>>> graph(n);
    for (auto& flight : flights)
        graph[flight[0]].emplace_back(flight[1], flight[2]);

    // Priority queue: {cost, current_node, stops_remaining}
    priority_queue<tuple<int, int, int>, vector<tuple<int, int, int>>, greater<>> pq;
    pq.emplace(0, src, k + 1);

    // Minimum cost to reach each node with specific stops remaining
    vector<vector<int>> minCost(n, vector<int>(k + 2, INT_MAX));
    minCost[src][k + 1] = 0;

    while (!pq.empty()) {
        auto [cost, node, stops] = pq.top();
        pq.pop();

        if (node == dst) return cost; // Destination reached with minimum cost
        if (stops > 0) {
            for (auto& [nextNode, price] : graph[node]) {
                int newCost = cost + price;
                if (newCost < minCost[nextNode][stops - 1]) {
                    minCost[nextNode][stops - 1] = newCost;
                    pq.emplace(newCost, nextNode, stops - 1);
                }
            }
        }
    }
    return -1; // No valid route found
}
```

### Explanation
1. **Graph Representation**:
   - Convert the flight data into an adjacency list `graph`, where each node points to its neighbors with the associated cost.

2. **Priority Queue**:
   - Use a min-heap (priority queue) to store the current cost, node, and remaining stops. This ensures exploration of cheaper paths first.

3. **Dynamic Tracking**:
   - Maintain a 2D vector `minCost[node][stops]` to record the minimum cost of reaching a node with a specific number of stops remaining.

4. **Early Termination**:
   - As soon as the destination node `dst` is dequeued from the priority queue, return its cost.

5. **Efficiency**:
   - Restricting the propagation by `k` ensures efficient traversal.
   - Priority queue ensures processing cheaper paths first.

### Complexity
- **Time Complexity**: 
  - \(O(E \cdot \log(E))\), where \(E\) is the number of edges. The priority queue operations dominate the complexity.
- **Space Complexity**: 
  - \(O(n \cdot (k + 1))\) for the `minCost` table and adjacency list.

### Example Usage
```cpp
int main() {
    int n = 4;
    vector<vector<int>> flights = {
        {0, 1, 100}, {1, 2, 100}, {2, 3, 100}, {0, 3, 500}
    };
    int src = 0, dst = 3, k = 1;

    int result = findCheapestPrice(n, flights, src, dst, k);
    // Output: 200
    return 0;
}
```

This implementation is efficient and provides the desired result within the problem constraints.
