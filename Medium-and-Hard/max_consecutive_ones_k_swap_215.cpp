class Solution 
{
public:
    int longestOnes(vector<int>& nums, int k) 
    {
        int zeros = 0; 
        int max_len = 0;
        int l = 0;

        for (int r = 0; r < nums.size(); r++) 
	{
            if (nums[r] == 0)
                zeros++;

            if(zeros>k)
	    {
                if(nums[l]==0)
                    zeros--;
                l++;
            }
            else
                max_len = max(max_len, r - l + 1); 
        }
        return max_len;
    }
};
