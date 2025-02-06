class Solution 
{
public:
    int networkDelayTime(vector<vector<int>>& times, int n, int k) 
    {
        
        vector<vector<pair<int, int>>> graph(n + 1);
        for (auto& t : times) 
	{
            int src = t[0], des = t[1], time = t[2];
            graph[src].push_back({des, time});
        }

       
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
        vector<int> dist(n + 1, INT_MAX);
        
        dist[k] = 0;
        pq.push({0, k}); // {time, node}

        while (!pq.empty()) 
	{
            auto [cost, node] = pq.top();
            pq.pop();

            if (cost > dist[node]) continue;
            
            for (auto& [neighbor, weight] : graph[node]) 
	    {
                if (cost + weight < dist[neighbor]) 
		{
                    dist[neighbor] = cost + weight;
                    pq.push({dist[neighbor], neighbor});
                }
            }
        }

        int ans = 0;
        for (int i = 1; i <= n; i++) 
	{
            if (dist[i] == INT_MAX) return -1;
            ans = max(ans, dist[i]);
        }

        return ans;
    }
};

//Code Explanation
Intuition
The problem can be solved using Dijkstra's Algorithm since we need to find the shortest path from a single source (k) to all nodes.

Graph Representation:

The input is given as a list of directed edges with weights.
We represent this as an adjacency list using a vector of pairs.
Dijkstra’s Algorithm:

We use a min-heap (priority queue) to always expand the node with the smallest current distance.
If we reach a node with a cost greater than the previously stored cost, we skip further processing.
Otherwise, we update the cost for that node and push it into the priority queue for further exploration.
Final Answer Calculation:

We determine the maximum shortest distance among all nodes.
If any node remains unreachable (INT_MAX), we return -1.
Approach
Build the Graph

Construct an adjacency list where graph[src] stores {destination, time} pairs.
Dijkstra’s Algorithm

Use a priority queue (min-heap) to process nodes in order of smallest current distance.
Start with node k having distance 0.
For each processed node, update the shortest distance to its neighbors and push them into the queue.
Calculate the Result

Find the maximum of all shortest distances (dist[i]).
If any node is unreachable (INT_MAX), return -1.
Complexity Analysis
Time Complexity:
O(E log V) → Dijkstra’s Algorithm using a Min-Heap
E is the number of edges, V is the number of nodes.
Each edge is processed once (O(E)) and inserted into the priority queue (O(log V)).
Space Complexity:
O(V + E) → Storing the graph as an adjacency list (O(E)), distance array (O(V)), and priority queue (O(V)).
