class Solution {
public:
    int triangleNumber(vector<int>& nums) 
    {
        int count = 0;
        int n = nums.size();
        
        // Sort the array first
        sort(nums.begin(), nums.end());
        
        // Iterate over the array, treating each element as the largest side (c)
        for (int k = n - 1; k >= 2; --k) 
	{
            int i = 0, j = k - 1;
            
            // Use two pointers to find valid (a, b) pairs
            while (i < j) 
	    {
                if (nums[i] + nums[j] > nums[k]) 
		{
                    // If nums[i] + nums[j] > nums[k], then all pairs (i, j), (i, j-1), ..., (i, i+1) are valid
                    count += (j - i);
                    --j;
                } 
		else 
		{
                    ++i;
                }
            }
        }
        
        return count;
    }
};
