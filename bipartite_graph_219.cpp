class Solution 
{
    public:
    bool Ans(vector<vector<int>>& graph, vector<int>& vis, int i)
    {
        queue<int> q;
        q.push(i);
        vis[i]=0;
        while(!q.empty())
	{
            int curr =q.front();
            q.pop();
            for(auto& v: graph[curr])
	    {
                if(vis[v]==-1)
		{
                    q.push(v);
                    vis[v] = !vis[curr];
                }
		else
		{
                    if(vis[v]==vis[curr]) return false;
                }
            }
        }
        return true;
    }
   
    bool isBipartite(vector<vector<int>>& graph) 
    {
        int n = graph.size();
        queue<int> q;
        vector<int> vis(n,-1);
        for(int i=0;i<n;i++)
	{
            if(vis[i]==-1)
	    {
                if(Ans(graph,vis,i)==false) return false;
            }
        }
        return true;
    }
};
