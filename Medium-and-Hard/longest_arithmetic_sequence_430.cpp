class Solution 
{
    public:
    int longestArithSeqLength(vector<int>& A) 
	{
        int n=A.size();
        unordered_map<int,unordered_map<int,int>>dp;
        
        int ans=0;
        for(int i=0;i<n;i++)
	{
            for(int j=0;j<i;j++)
	    {
                int d=A[i]-A[j];
                if(dp[d].find(j)==dp[d].end())
		{
                    dp[d][i]=2;
                }
                else
		{
		    dp[d][i]=max(dp[d][j]+1,dp[d][i]);
		}
                ans=max(ans,dp[d][i]);
            }
        }
        return ans;
    }
};
