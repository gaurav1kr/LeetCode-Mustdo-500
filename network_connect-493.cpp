#include<vector>
#include<iostream>
using namespace std;
/* The best way is to find no of disconnected components and reduce the total by 1*/
class Solution
{
public:
    void dfs(vector<int>& visited, vector<int>* adj_list, int i)
    {
        for (auto& n : adj_list[i])
            if (!visited[n])
            {
                visited[n] = 1;
                dfs(visited, adj_list, n);
            }
    }
    int makeConnected(int n, vector<vector<int>>& connections)
    {
        int no_of_edges = connections.size();
        int i = 0;
        int disconnected = 0;
        if (no_of_edges < n - 1)
        {
            return -1;
        }
        vector<int> visited(n, 0);
        vector<int> adj_list[n];
        for (auto& conn : connections)
        {
            adj_list[conn[0]].push_back(conn[1]);
            adj_list[conn[1]].push_back(conn[0]);
        }

        for (i = 0; i < n; i++)
        {
            if (!visited[i])
                disconnected++;

            visited[i] = 1;
            dfs(visited, adj_list, i);
        }
        return disconnected - 1;
    }
};