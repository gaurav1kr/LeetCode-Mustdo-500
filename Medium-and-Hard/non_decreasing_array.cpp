class Solution 
{
	public:
    	bool checkPossibility(vector<int>& nums) 
	{
        	const int n = nums.size();
        	int drop_count = 0;
        	for (int i = 0; i < n - 1; i++) 
		{
            		if (nums[i] > nums[i + 1]) 
			{
                		drop_count++;
                		if (drop_count > 1 or (i > 0 and i < n - 2 and nums[i - 1] > nums[i + 1] and nums[i] > nums[i + 2])) 
				{
                    			return false;
                		}
            		}
        	}
        	return true;
    	}
};
