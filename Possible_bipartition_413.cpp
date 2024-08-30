class Solution 
{
    private:
		void solve(vector<vector<int>>& adj ,int src, vector<int>& vis,bool &ans , int color)
		{
			vis[src] = color;
			for(auto a : adj[src])
			{
				if(vis[a]==-1)
				{
					solve(adj,a,vis,ans,1-color);
				}
				else
				{
					if(vis[a]==color)
					{
						ans=false;
						return;
					}
				}
			}

		}
	public:
		bool possibleBipartition(int n, vector<vector<int>>& dislikes) 
		{
		   vector<vector<int>> adj(n+1);
		   for(int i=0; i<dislikes.size(); i++)
		   {
			   int u = dislikes[i][0];
			   int v = dislikes[i][1];
			   adj[u].push_back(v);
			   adj[v].push_back(u);
		   }

		   vector<int>vis(n+1,-1);
		   bool ans = true;
		   for(int i=1;i<=n;i++)
		   {
			   if(vis[i]==-1)
			   {
				   solve(adj,i,vis,ans,0);
			   }
		   }
		   return ans;
		}
};