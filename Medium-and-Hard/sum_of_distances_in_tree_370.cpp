class Solution 
{
public:
    vector<int> res; // to keep track distance of each node
    vector<int> count;
    void fun(int node, int parent, vector<int> adj[]) 
	{//Pre-order to calculate left and right child
        for (int it : adj[node]) 
		{
            if (it != parent) {
                fun(it, node, adj);
                count[node] += count[it];
                res[node] += res[it] + count[it];
            }
        }
    }
        //pos-Order to calculate root dis from left and right
    void fun2(int node, int parent, vector<int> adj[],int n) 
	{
        for (int it : adj[node]) 
		{
            if (parent != it) 
			{
                res[it] = res[node] - count[it] + (n - count[it]);
                fun2(it, node, adj,n);
            }
        }
    }

    vector<int> sumOfDistancesInTree(int n, vector<vector<int>>& edges) 
	{
        int sum = 0;
        int m = edges.size();
        vector<int> adj[n];
        count = vector<int>(n, 1);
        res = vector<int>(n, 0);
        for (auto& it : edges) 
		{
            int u = it[0];
            int v = it[1];
            adj[u].push_back(v);
            adj[v].push_back(u);
        }
        fun(0, -1, adj);
        fun2(0, -1, adj,n);

        return res;
    }
};