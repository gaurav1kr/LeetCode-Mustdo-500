class Solution {
public:
    int shortestPath(vector<vector<int>>& grid, int k) 
	{
            queue<pair<pair<int,int>, pair<int,int>>> q;
            int n = grid.size(), m = grid[0].size();
            int dp[n][m][k+1];
            memset(dp, 0, sizeof dp);
            dp[0][0][k] = 1;
            int dx[4] = {-1, 0, 1, 0};
            int dy[4] = {0, -1, 0, 1};
            q.push({{0, 0}, {k, 0}});
            while(!q.empty()){
            int x = q.front().first.first;
            int y = q.front().first.second;
            int rem = q.front().second.first;
            int moves = q.front().second.second;
            q.pop();
            if(x==n-1 && y==m-1)    
	    {
	    	return moves;
	    }
            for(int i = 0; i < 4; i++)
	    {
                int nx = x + dx[i];
                int ny = y + dy[i];
                if(nx<0 || ny<0 || nx==n || ny==m)  continue;
                if(grid[nx][ny] == 1)
		{
                    if(rem == 0)    
		    {
		    	continue;
		    }
                    else
		    {
                        if(!dp[nx][ny][rem-1])
			{
                            q.push({{nx, ny}, {rem-1, moves+1}});
                            dp[nx][ny][rem-1] = 1;
                        }
                    }
                }
		else
		{
                    if(!dp[nx][ny][rem])
		    {
                        q.push({{nx, ny}, {rem, moves+1}});
                        dp[nx][ny][rem] = 1;
                    }
                }
            }
        }
        return -1;
    }
};
