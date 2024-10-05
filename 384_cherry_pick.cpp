class Solution 
{
public:
    int solve(vector<vector<int>>& grid,int r1,int c1,int r2,int c2,int n,vector<vector<vector<int>>>& dp)
    {
        int cherries=0;
        if(r1>=n||r2>=n||c1>=n||c2>=n||grid[r1][c1]==-1||grid[r2][c2]==-1)
        {
            return INT_MIN;
        }
        if(r1==n-1&&c1==n-1) 
        {
            return grid[r1][c1];
        }
        if(dp[r1][c1][r2]!=-1) 
        {
            return dp[r1][c1][r2];
        }
       
        if(r1==r2&&c1==c2)
        {
            cherries+=grid[r1][c1];
        }
        else
        {
            cherries+=grid[r1][c1];
            cherries+=grid[r2][c2];
        }
        int f1=solve(grid,r1,c1+1,r2,c2+1,n,dp);
        int f2=solve(grid,r1+1,c1,r2,c2+1,n,dp);
        int f3=solve(grid,r1+1,c1,r2+1,c2,n,dp);
        int f4=solve(grid,r1,c1+1,r2+1,c2,n,dp);
        cherries+=max(max(f1,f2),max(f3,f4));
        return dp[r1][c1][r2]=cherries;
    }
    int cherryPickup(vector<vector<int>>& grid) 
    {
        int n=grid.size();
        int r1=0,c1=0,r2=0,c2=0;
        vector<vector<vector<int>>> dp(n+1, vector<vector<int>>(n+1, vector<int>(n+1, -1)));
        int ans=solve(grid,r1,c1,r2,c2,n,dp);
        return max(0,ans);
    }
}; 