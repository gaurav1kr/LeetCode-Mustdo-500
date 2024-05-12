class Solution 
{
public:
    int rob(vector<int>& nums) 
    {
        if(nums.size() == 0) return 0;
        if(nums.size() == 1) return nums[0];
        int cache[2];
        cache[0] = nums[0];
        cache[1] = max(nums[0], nums[1]);
        for(int i = 2; i < nums.size(); i++) 
        {
            cache[i & 1] = max(cache[i&1]+nums[i], cache[(i-1)&1]);
        }
        return max(cache[0], cache[1]);
    }
};
