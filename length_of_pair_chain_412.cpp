class Solution 
{
public:
    int dp[2002];
    int end_to_min_len[2002];
    int findLongestChain(vector<vector<int>>& pairs) 
	{
        memset(dp, 0, sizeof(dp));
        memset(end_to_min_len, 0, sizeof(end_to_min_len));
        int pmax = -1001;
        int pmin = 1001;
        for(const auto& p: pairs)
		{
            end_to_min_len[p[1]+1000] = end_to_min_len[p[1]+1000] == 0 ? p[1]-p[0] :
            min(end_to_min_len[p[1]+1000], p[1]-p[0]);
            pmin = min(p[0], pmin);
            pmax = max(p[1], pmax);
        }
        for(int i = pmin; i <= pmax; i++)
		{
            dp[i+1001] = dp[i+1000];
            if(end_to_min_len[i+1000])
			{
                dp[i+1001] = max(dp[i+1001], dp[i+1000-end_to_min_len[i+1000]]+1);
            }
        }
        return dp[pmax+1001];
    }
};