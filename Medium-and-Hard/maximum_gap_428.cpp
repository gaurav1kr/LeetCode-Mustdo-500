class Solution 
{
public:
    int maximumGap(vector<int>& nums) 
    {
        int size_nums = nums.size() ;
        sort(nums.begin() , nums.end()) ;
        int max = INT_MIN ;
        for(int i=0;i<size_nums-1;i++)
        {
            max = (nums[i+1] - nums[i]) > max ?  (nums[i+1] - nums[i]) : max ;
        }
        return (max==INT_MIN)?0:max ;
    }
};
