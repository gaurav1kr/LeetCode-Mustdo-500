class Solution 
{
public:
    int findNumberOfLIS(vector<int>& nums) 
	{
        if (nums.empty()) 
			return 0;
        
        int n = nums.size();
        vector<int> dp(n, 1);  // dp[i] will store the length of LIS ending at i
        vector<int> count(n, 1);  // count[i] will store the number of LIS ending at i
        
        int maxLength = 1;
        int totalCount = 0;
        
        for (int i = 1; i < n; ++i) 
		{
            for (int j = 0; j < i; ++j) 
			{
                if (nums[i] > nums[j]) 
				{
                    if (dp[j] + 1 > dp[i]) 
					{
                        dp[i] = dp[j] + 1;
                        count[i] = count[j];  // reset the count to count[j]
                    } 
					else if (dp[j] + 1 == dp[i]) 
					{
                        count[i] += count[j];  // add count[j] to count[i]
                    }
                }
            }
            maxLength = max(maxLength, dp[i]);  // update maxLength
        }
        
        for (int i = 0; i < n; ++i) 
		{
            if (dp[i] == maxLength) 
			{
                totalCount += count[i];
            }
        }
        
        return totalCount;
    }
};
