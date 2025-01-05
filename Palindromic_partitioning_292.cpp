class Solution 
{
public:
    int solve(int i,string &s, vector<int> &dp)
	{
        if(i==s.size())
		{
            return 0;
        }
        if(dp[i]!=INT_MAX)return dp[i];
        
        for(int j=i;j<s.size();j++)
		{
            if(isPalin(s,i,j))
			{
                dp[i]=min(dp[i],solve(j+1,s,dp)+1);
            }
        }
        return dp[i];
    }

    bool isPalin(string &s, int start,int end)
	{
        while(start<=end)
		{
            if(s[start++]!=s[end--])
				return false;
        }
        return true;
    }
    int minCut(string s) 
	{
        vector<int> dp(s.size()+1,INT_MAX);
        return solve(0,s,dp)-1;
    }
};