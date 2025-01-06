class Solution 
{
public:
	unordered_map<int,int>mp;

	int deleteAndEarn(vector<int>& nums)
     {
		int n=nums.size();
		sort(nums.begin(),nums.end());
		for(auto i:nums) mp[i]++;
		vector<int> dp(n+1,0);
		for(int i=n-1;i>=0;i--)
        {
			int del=mp[nums[i]]*nums[i]+dp[upper_bound(nums.begin(),nums.end(),nums[i]+1)-nums.begin()];
			int notdel=dp[i+1];
			dp[i]=max(del,notdel);
		}
		return dp[0];
	}
};