class Solution 
{
public:
    int maxProfit(vector<int>& p, int fee) 
    {
        int n = p.size();
        vector<vector<int>> dp(n+1, vector<int>(2, 0));

        for(int i = n-1; i>=0; i--)
        for(int hold = 0; hold<=1; hold++)
        {
                if(hold)
                    dp[i][hold] = max( p[i] + dp[i+1][0] - fee, dp[i+1][1] );
                else
                    dp[i][hold] = max( -p[i] + dp[i+1][1], dp[i+1][0] );
        }
        return dp[0][0];
    }
};
