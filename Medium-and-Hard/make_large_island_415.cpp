class DisjointSet
{
    public:
    vector<int>rank,parent,size;
    DisjointSet(int n)
	{
        rank.resize(n+1,0);
        parent.resize(n+1);
        size.resize(n+1);
        for(int i=0;i<=n;i++)
		{
            parent[i]=i;
            size[i]=1;
        } 
    }

    int findUPar(int n)
	{
        if(parent[n]==n) return n;
        return parent[n]=findUPar(parent[n]);
    }

    void unionBySize(int u,int v)
	{
        int ulp_u=findUPar(u);
        int ulp_v=findUPar(v);
        if(ulp_u==ulp_v) return ;
        if(size[ulp_u]<size[ulp_v])
		{
            parent[ulp_u]=ulp_v;
            size[ulp_v]+=size[ulp_u];
        } 
        else
		{
            parent[ulp_v]=ulp_u;
            size[ulp_u]+=size[ulp_v];
        }
    }
};

class Solution 
{
public:
    int largestIsland(vector<vector<int>>& grid) 
	{
        int n=grid.size();
        DisjointSet ds(n*n);
        int x[]={-1,0,1,0};
        int y[]={0,1,0,-1};
        for(int i=0;i<n;i++)
		{
            for(int j=0;j<n;j++)
			{
                if(grid[i][j]==1)
				{
                    for(int k=0;k<4;k++)
					{
                        int ni=i+x[k];
                        int nj=j+y[k];
                        if(ni>=0 && nj>=0 && ni<n && nj<n && grid[ni][nj]==1)
						{
                            int adjNode=n*ni+nj;
                            int parNode=n*i+j;
                            if(ds.findUPar(parNode)!=ds.findUPar(adjNode))
							{
                                ds.unionBySize(parNode,adjNode);
                            }
                        }
                    }
                }
            }
        }

        int count=0;
        for(int i=0;i<n;i++){
            for(int j=0;j<n;j++)
			{
                if( grid[i][j]==0)
				{
                    set<int>s;
                    for(int k=0;k<4;k++)
					{
                        int ni=i+x[k];
                        int nj=j+y[k];
                        if(ni>=0 && nj>=0 && ni<n && nj<n && grid[ni][nj]==1)
						{
                            int adjNode=n*ni+nj;
                            int parAdjNode=ds.findUPar(adjNode);
                            s.insert(parAdjNode);
                        }
                    }
                   int totalSize=0;
                   for(auto x:s)
				   {
						totalSize+=ds.size[x];
                   } 
                 count=max(count,totalSize+1);
                }
            }
        }
        
        for(int i=0;i<n*n;i++)
		{
            count=max(count,ds.size[ds.findUPar(i)]);
        }
        return count;
    }
};